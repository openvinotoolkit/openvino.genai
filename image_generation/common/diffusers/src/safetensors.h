/*
        Harris M. Snyder, 2023
        This is free and unencumbered software released into the public domain.

        safetensors.h: a library for reading .safetensors files from C.

        Basic usage: read the entire .safetensors file into memory (this is not
        handled by safetensors.h) and feed it to safetensors_file_init(). This
        will populate a safetensors_File struct, which contains an array of
        tensor descriptors. You can then loop over the tensor descriptors and
        pull out what you need. See the structs and functions below for details.

        This file is a single-header library (credit to Sean Barrett for the
        idea); it includes both the header and the actual definitions in
        a single file. To use this library, copy it  into your project, and
        define SAFETENSORS_IMPLEMENTATION in exactly one .c file, immediately
        before you include safetensors.h

        The library depends only on the following headers from the standard
        library:
          -  limits.h
          -  stdint.h
          -  stdlib.h
        The latter is for realloc. A future update will allow the user to
        control the memory allocation, so that stdlib.h is not needed.

*/

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>

#ifndef SAFETENSORS_MAX_DIM
#    define SAFETENSORS_MAX_DIM 20
#endif

typedef struct {
    int len;
    char* ptr;
} safetensors_Str;

typedef struct {
    safetensors_Str name;
    // the pointer inside this struct will point into the
    // memory block passed to safetensors_file_init()

    int dtype;
    // will be one of the enum values below

    int n_dimensions;
    int64_t shape[SAFETENSORS_MAX_DIM];
    // only the first n_dimensions entry of shape are meaningful

    int64_t begin_offset_bytes;
    int64_t end_offset_bytes;
    // values taken directly from file. an offset of 0 means the
    // exact start of the portion of the file that follows the
    // header (i.e. it is NOT an offset into the entire file).

    void* ptr;
    // this will be pre-populated assuming that the memory block
    // that was fed to safetensors_file_init() was the entire file,
    // i.e. that the actual data immediately follows the header.
    // if this is not the case, this pointer will be bogus, use the
    // offsets to manually compute the location.
} safetensors_TensorDescriptor;

typedef struct {
    safetensors_Str name;
    safetensors_Str value;
} safetensors_MetadataEntry;

typedef struct {
    int c;  // internal use

    char* error_context;
    // if safetensors_file_init() fails, this pointer will be set to
    // where in the file memory block the error occurred.

    void* one_byte_past_end_of_header;
    // after calling safetensors_file_init, this will point to the
    // next byte after the end of the header

    safetensors_TensorDescriptor* tensors;
    safetensors_MetadataEntry* metadata;
    // these ^ are allocated automatically by safetensors_file_init()
    // and are contiguous arrays. the user should free() them when done.

    int num_tensors;
    int num_metadata;
    // the lengths of the above arrays
} safetensors_File;

char* safetensors_file_init(void* file_buffer, int64_t file_buffer_size_bytes, safetensors_File* out);
// Given a file buffer, parses the safetensors header and populates a safetensors_File
// structure so that the client program can find the data it wants. file_buffer should
// point to a buffer that contains at least the entire header, or more preferably the
// whole safetensors file.
//
// Returns 0 on success. On failure, returns a static error message string and sets
// out->error_context such that it points to where in file_buffer the error happened.

static int safetensors_str_equal(safetensors_Str a, const char* b)
// For convenience: easily check if a tensor name matches a given string literal
{
    if (!b)
        return 0;
    int equal = 1;
    for (int i = 0; (i < a.len && equal && b[i]); i++)
        equal = equal && a.ptr[i] == b[i];
    return equal;
}

// Enum values for the 'dtype' field
enum {
    SAFETENSORS_F64 = 0,
    SAFETENSORS_F32,
    SAFETENSORS_F16,
    SAFETENSORS_BF16,
    SAFETENSORS_I64,
    SAFETENSORS_I32,
    SAFETENSORS_I16,
    SAFETENSORS_I8,
    SAFETENSORS_U8,
    SAFETENSORS_BOOL,

    SAFETENSORS_NUM_DTYPES
};

#endif

/*
   ============================================================================
                END OF HEADER SECTION
                Implementation follows
   ============================================================================
*/

#ifdef SAFETENSORS_IMPLEMENTATION

#ifndef assert
#    ifdef SAFETENSORS_DISABLE_ASSERTIONS
#        define assert(c)
#    else
#        if defined(_MSC_VER)
#            define assert(c)       \
                if (!(c)) {         \
                    __debugbreak(); \
                }
#        else
#            if defined(__GNUC__) || defined(__clang__)
#                define assert(c)         \
                    if (!(c)) {           \
                        __builtin_trap(); \
                    }
#            else
#                define assert(c)              \
                    if (!(c)) {                \
                        *(volatile int*)0 = 0; \
                    }
#            endif
#        endif
#    endif
#endif

#include <limits.h>
#include <stdlib.h>

static int64_t parse_positive_int(char** ptr, char* limit) {
    /*
            - Skips preceeding spaces and tabs
            - Won't read past 'limit'
            - Doesn't check for integer overflow
            - Doesn't parse negative numbers
            - Returns -1 on failure
    */
    char* str = *ptr;

    while (*str == ' ' || *str == '\t')
        str++;

    int64_t v = 0;
    int n = 0;
    while (str < limit && *str >= 48 && *str <= 57) {
        int digit = *str - 48;
        v *= 10;
        v += digit;
        str++;
        n++;
    }

    if (n > 0) {
        *ptr = str;
        return v;
    }

    return -1;
}

static int eat(char** ptr, char* limit, char expected) {
    // skip whitespace
    // return 0 if we hit the limit
    // return 0 if data don't match the expected string
    // otherwise, return 1 and move the pointer to the end of the match
    char* p = *ptr;
    while (*p == ' ' || *p == '\t')
        ++p;
    if (p + 1 > limit)
        return 0;
    if (*p != expected)
        return 0;
    *ptr = p + 1;
    return 1;
}

static int peek(char* ptr, char* limit, char expected) {
    // same as eat, but doesn't adjust the pointer
    char* tmp = ptr;
    return eat(&tmp, limit, expected);
}

typedef struct {
    int num_entries;
    int64_t entries[SAFETENSORS_MAX_DIM];
} IntList;

static int eat_intlist(char** ptr, char* limit, IntList* out) {
    // *out = (IntList){0};
    *out = IntList();
    out->num_entries = 0;
    char* p = *ptr;
    if (!eat(&p, limit, '['))
        return 0;

    while (p < limit) {
        char* p_save = p;
        if (eat(&p, limit, ']'))
            break;

        int64_t val = parse_positive_int(&p, limit);
        if (val == -1) {
            return 0;
        } else {
            out->entries[out->num_entries++] = val;
            if (out->num_entries == SAFETENSORS_MAX_DIM) {
                return 0;  // unsupported tensor dimensions (TODO improve handling)
            }
        }

        if (!eat(&p, limit, ','))
            if (!peek(p, limit, ']'))
                return 0;

        assert(p != p_save);
    }

    *ptr = p;
    return 1;
}

static int eat_string(char** ptr, char* limit, safetensors_Str* out) {
    char delim = 0;

    if (eat(ptr, limit, '\''))
        delim = '\'';
    else if (eat(ptr, limit, '"'))
        delim = '"';
    else
        return 0;  // bad delimiter

    int len = 0;
    char* p = *ptr;
    char* start = p;

    while (p < limit) {
        if (*p == delim && p[-1] != '\\') {
            ++p;
            goto string_ok;
        } else {
            ++p;
            ++len;
        }
    }
    return 0;  // unterminated

string_ok:
    assert(p <= limit);
    *ptr = p;
    // *out = (safetensors_Str) { .len=len, .ptr=start};
    safetensors_Str str;
    str.len = len;
    str.ptr = start;
    *out = str;
    return 1;
}

typedef struct {
    safetensors_Str key;
    int value_is_str;
    union {
        safetensors_Str svalue;
        IntList ivalue;
    };
} KeyValuePair;

static int eat_kv_pair(char** ptr, char* limit, KeyValuePair* kvp) {
    char* p = *ptr;

    // mandatory string (key)
    if (!eat_string(&p, limit, &kvp->key))
        return 0;

    if (!eat(&p, limit, ':'))
        return 0;

    // value can be string, or list of integers
    safetensors_Str str_value = {0};
    IntList intlist_value = {0};

    if (!eat_string(&p, limit, &str_value)) {
        if (!eat_intlist(&p, limit, &intlist_value)) {
            return 0;
        } else {
            kvp->value_is_str = 0;
            kvp->ivalue = intlist_value;
        }
    } else {
        kvp->value_is_str = 1;
        kvp->svalue = str_value;
    }

    *ptr = p;
    return 1;
}

static void mem_copy(void* dest, void* source, unsigned num) {
    // unsigned char *d=dest, *s=source;
    unsigned char* d = (unsigned char*)dest;
    unsigned char* s = (unsigned char*)source;
    for (unsigned i = 0; i < num; i++)
        d[i] = s[i];
}

static char* more_memory(safetensors_File* out) {
    if (out->num_tensors == out->c || out->num_metadata == out->c) {
        void* new_tensors = realloc(out->tensors, sizeof(out->tensors[0]) * (out->c + 100));
        // if (!new_tensors) return "Out of memory";
        if (!new_tensors)
            return const_cast<char*>("Out of memory");

        // out->tensors  = new_tensors;
        out->tensors = (safetensors_TensorDescriptor*)new_tensors;

        void* new_metadata = realloc(out->metadata, sizeof(out->metadata[0]) * (out->c + 100));
        // if (!new_metadata) return "Out of memory";
        if (!new_metadata)
            return const_cast<char*>("Out of memory");

        // out->metadata = new_metadata;
        out->metadata = (safetensors_MetadataEntry*)new_metadata;
        out->c += 100;
    }
    return 0;
}

char* apply_key_value_pair(safetensors_File* out, KeyValuePair kvp, char* baseptr) {
    // #define KNOWN_DTYPES "F64, F32, F16, BF16, I64, I32, I16, I8, U8, or BOOL"
    if (safetensors_str_equal(kvp.key, "dtype")) {
        if (!kvp.value_is_str)
            return const_cast<char*>("Expected a string value for 'dtype'");
        if (safetensors_str_equal(kvp.svalue, "F64"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_F64;
        else if (safetensors_str_equal(kvp.svalue, "F32"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_F32;
        else if (safetensors_str_equal(kvp.svalue, "F16"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_F16;
        else if (safetensors_str_equal(kvp.svalue, "BF16"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_BF16;
        else if (safetensors_str_equal(kvp.svalue, "I64"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_I64;
        else if (safetensors_str_equal(kvp.svalue, "I32"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_I32;
        else if (safetensors_str_equal(kvp.svalue, "I16"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_I16;
        else if (safetensors_str_equal(kvp.svalue, "I8"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_I8;
        else if (safetensors_str_equal(kvp.svalue, "U8"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_U8;
        else if (safetensors_str_equal(kvp.svalue, "BOOL"))
            out->tensors[out->num_tensors].dtype = SAFETENSORS_BOOL;
        // else return "Unrecognized datatype (expected " KNOWN_DTYPES ")";
        else
            return const_cast<char*>(
                "Unrecognized datatype (expected F64, F32, F16, BF16, I64, I32, I16, I8, U8, or BOOL)");

    } else if (safetensors_str_equal(kvp.key, "shape")) {
        if (kvp.value_is_str)
            return const_cast<char*>("Expected an integer list value for 'shape'");
        out->tensors[out->num_tensors].n_dimensions = kvp.ivalue.num_entries;
        for (int i = 0; i < kvp.ivalue.num_entries; i++)
            out->tensors[out->num_tensors].shape[i] = kvp.ivalue.entries[i];
    } else if (safetensors_str_equal(kvp.key, "data_offsets")) {
        if (kvp.value_is_str)
            return const_cast<char*>("Expected an integer list value for 'shape'");
        if (kvp.ivalue.num_entries != 2)
            return const_cast<char*>("Expected exactly two entries for the value of 'offsets'");
        out->tensors[out->num_tensors].begin_offset_bytes = kvp.ivalue.entries[0];
        out->tensors[out->num_tensors].end_offset_bytes = kvp.ivalue.entries[1];
        out->tensors[out->num_tensors].ptr = baseptr + kvp.ivalue.entries[0];
    } else {
        // error? ignore?
        return const_cast<char*>("Unexpected key (expected dtype, shape, or data_offsets)");
    }
    return 0;
}

char* safetensors_file_init(void* file_buffer, int64_t file_buffer_bytes, safetensors_File* out) {
    // *out = (safetensors_File){0};
    safetensors_File file;
    file.c = 0;
    file.error_context = nullptr;
    file.one_byte_past_end_of_header = nullptr;
    file.tensors = nullptr;
    file.metadata = nullptr;
    file.num_tensors = 0;
    file.num_metadata = 0;

    *out = file;
    int header_len = 0;
    {
        uint64_t header_len_u64 = 0;
        mem_copy(&header_len_u64, file_buffer, sizeof(header_len_u64));
        // if (header_len_u64 > (uint64_t)INT_MAX)
        if (header_len_u64 > static_cast<uint64_t>(INT_MAX)) {
#define STRINGIFY(x) #x
            // 	return "File header allegedly more than INT_MAX (" STRINGIFY(INT_MAX) ") bytes, file likely corrupt";
            return const_cast<char*>(
                "File header allegedly more than INT_MAX (" STRINGIFY(INT_MAX) ") bytes, file likely corrupt");
        }
        header_len = header_len_u64;
    }
    assert(header_len >= 0);
    if (header_len == 0)
        return const_cast<char*>("File header allegedly zero bytes, file likely corrupt");

    char* t = ((char*)file_buffer) + 8;
    char* e = t + header_len;
    out->one_byte_past_end_of_header = e;

    char* tensor_data_baseptr = t + header_len;

// #define ST_ERR(message) return out->error_context=t, (message);
#define ST_ERR(message) return out->error_context = t, const_cast<char*>(message)
    // mandatory open brace starts the header
    if (!eat(&t, e, '{'))
        ST_ERR("Expected '{'");

    // loop over header entries
    while (t < e) {
        char* t_save = t;

        // if we hit a close brace, we're done
        if (eat(&t, e, '}'))
            goto header_ok;

        // mandatory string (tensor name)
        safetensors_Str tensor_name = {0};
        if (!eat_string(&t, e, &tensor_name))
            ST_ERR("Expected tensor name");
        if (!eat(&t, e, ':'))
            ST_ERR("Expected colon after tensor name");

        char* alloc_error = more_memory(out);
        if (alloc_error)
            ST_ERR(alloc_error);

        out->tensors[out->num_tensors].name = tensor_name;

        // open brace starts a header entry
        if (eat(&t, e, '{')) {
            // loop over key-value pairs inside the header entry
            while (t < e) {
                char* t_save = 0;

                // close brace terminates the header entry
                if (eat(&t, e, '}')) {
                    if (!safetensors_str_equal(tensor_name, "__metadata__"))
                        ++out->num_tensors;
                    break;
                }

                // otherwise it's a key-value pair
                KeyValuePair kvp = {0};
                char* error_context = t;
                if (!eat_kv_pair(&t, e, &kvp))
                    ST_ERR("Expected a key-value pair");

                // figure out what to do with the key-value pair
                if (safetensors_str_equal(tensor_name, "__metadata__")) {
                    if (!kvp.value_is_str)
                        return out->error_context = error_context,
                               const_cast<char*>("Expected a string value for a metadata entry");
                    // out->metadata[out->num_metadata++] =
                    // 	(safetensors_MetadataEntry) {
                    // 		.name  = kvp.key,
                    // 		.value = kvp.svalue
                    // 	};
                    safetensors_MetadataEntry entry;
                    entry.name = kvp.key;
                    entry.value = kvp.svalue;
                    out->metadata[out->num_metadata++] = entry;
                } else {
                    char* kvp_error = apply_key_value_pair(out, kvp, tensor_data_baseptr);
                    if (kvp_error)
                        return out->error_context = error_context, kvp_error;
                }

                if (!eat(&t, e, ','))
                    if (!peek(t, e, '}'))
                        ST_ERR("Expected comma");

                assert(t != t_save);
            }
        }

        if (!eat(&t, e, ','))
            if (!peek(t, e, '}'))
                ST_ERR("Expected comma");

        assert(t != t_save);
    }
    ST_ERR("Unterminated header");
header_ok:
    return 0;
#undef ST_ERR
}

#endif
