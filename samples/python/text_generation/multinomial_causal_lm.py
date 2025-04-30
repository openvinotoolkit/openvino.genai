#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import queue
import threading
from typing import Union


class IterableStreamer(openvino_genai.StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
        tokens_cache (list): A buffer to accumulate tokens for detokenization.
        text_queue (Queue): A synchronized queue for storing decoded text chunks.
        print_len (int): The length of the printed text to manage incremental decoding.
    """

    def __init__(self, tokenizer):
        """
        Initializes the IterableStreamer with the given tokenizer.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self.decoded_lengths = []

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next value from the text queue.

        Returns:
            str: The next decoded text chunk.

        Raises:
            StopIteration: If there are no more elements in the queue.
        """
        # get() will be blocked until a token is available.
        value = self.text_queue.get()
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped or cancelled.

        Returns:
            openvino_genai.StreamingStatus: Always returns RUNNING in this implementation.
        """
        return openvino_genai.StreamingStatus.RUNNING

    def write_word(self, word: str):
        """
        Puts a word into the text queue.

        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def write(self, token: Union[int, list[int]]) -> openvino_genai.StreamingStatus:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.

        Args:
            token (Union[int, list[int]]): The token(s) to process.

        Returns:
            bool: True if generation should be stopped, False otherwise.
        """
        if type(token) is list:
            self.tokens_cache += token
            self.decoded_lengths += [-2 for _ in range(len(token) - 1)]
        else:
            self.tokens_cache.append(token)

        text = self.tokenizer.decode(self.tokens_cache)
        self.decoded_lengths.append(len(text))

        word = ""
        delay_n_tokens = 3
        if len(text) > self.print_len and "\n" == text[-1]:
            # Flush the cache after the new line symbol.
            word = text[self.print_len :]
            self.tokens_cache = []
            self.decoded_lengths = []
            self.print_len = 0
        elif len(text) > 0 and text[-1] == chr(65533):
            # Don't print incomplete text.
            self.decoded_lengths[-1] = -1
        elif len(self.tokens_cache) >= delay_n_tokens:
            self.compute_decoded_length_for_position(
                len(self.decoded_lengths) - delay_n_tokens
            )
            print_until = self.decoded_lengths[-delay_n_tokens]
            if print_until != -1 and print_until > self.print_len:
                # It is possible to have a shorter text after adding new token.
                # Print to output only if text length is increased and text is complete (print_until != -1).
                word = text[self.print_len : print_until]
                self.print_len = print_until
        self.write_word(word)

        stop_flag = self.get_stop_flag()
        if stop_flag != openvino_genai.StreamingStatus.RUNNING:
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()

        return stop_flag

    def compute_decoded_length_for_position(self, cache_position: int):
        # decode was performed for this position, skippping
        if self.decoded_lengths[cache_position] != -2:
            return

        cache_for_position = self.tokens_cache[: cache_position + 1]
        text_for_position = self.tokenizer.decode(cache_for_position)

        if len(text_for_position) > 0 and text_for_position[-1] == chr(65533):
            # Mark text as incomplete
            self.decoded_lengths[cache_position] = -1
        else:
            self.decoded_lengths[cache_position] = len(text_for_position)

    def end(self):
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len :]
            self.write_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.text_queue.put(None)


class ChunkStreamer(IterableStreamer):

    def __init__(self, tokenizer, tokens_len):
        super().__init__(tokenizer)
        self.tokens_len = tokens_len

    def write(self, token: Union[int, list[int]]) -> openvino_genai.StreamingStatus:
        if (len(self.tokens_cache) + 1) % self.tokens_len == 0:
            return super().write(token)

        if type(token) is list:
            self.tokens_cache += token
            # -2 means no decode was done for this token position
            self.decoded_lengths += [-2 for _ in range(len(token))]
        else:
            self.tokens_cache.append(token)
            self.decoded_lengths.append(-2)

        return openvino_genai.StreamingStatus.RUNNING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("prompt")
    args = parser.parse_args()

    device = "CPU"  # GPU can be used as well
    tokens_len = 10  # chunk size
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    text_print_streamer = ChunkStreamer(pipe.get_tokenizer(), tokens_len)

    def token_printer():
        # Getting next elements from iterable will be blocked until a new token is available.
        for word in text_print_streamer:
            print(word, end="", flush=True)

    printer_thread = threading.Thread(target=token_printer, daemon=True)
    printer_thread.start()

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.do_sample = True
    config.top_p = 0.9
    config.top_k = 30

    # Since the streamer is set, the results will be printed
    # every time a new token is generated and put into the streamer queue.
    pipe.generate(args.prompt, config, text_print_streamer)
    printer_thread.join()


if "__main__" == __name__:
    main()
