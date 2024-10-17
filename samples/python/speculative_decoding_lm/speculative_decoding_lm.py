#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import queue
import threading


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
        value = self.text_queue.get()  # get() will be blocked until a token is available.
        if value is None:
            raise StopIteration
        return value
    
    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped.
        
        Returns:
            bool: Always returns False in this implementation.
        """
        return False
    
    def put_word(self, word: str):
        """
        Puts a word into the text queue.
        
        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def put(self, token_id: int) -> bool:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.
        
        Args:
            token_id (int): The token_id to process.
        
        Returns:
            bool: True if generation should be stopped, False otherwise.
        """        
        self.tokens_cache.append(token_id)
        text = self.tokenizer.decode(self.tokens_cache)

        word = ''
        if len(text) > self.print_len and '\n' == text[-1]:
            # Flush the cache after the new line symbol.
            word = text[self.print_len:]            
            self.tokens_cache = []
            self.print_len = 0
        elif len(text) >= 3 and text[-3:] == chr(65533):
            # Don't print incomplete text.
            pass
        elif len(text) > self.print_len:
            # It is possible to have a shorter text after adding new token.
            # Print to output only if text lengh is increaesed.
            word = text[self.print_len:]
            self.print_len = len(text)
        self.put_word(word)        
        
        if self.get_stop_flag():
            # When generation is stopped from streamer then end is not called, need to call it here manually.
            self.end()
            return True  # True means stop  generation
        else:
            return False  # False means continue generation

    def end(self):
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len:]
            self.put_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.put_word(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('draft_model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well

    scheduler_config = openvino_genai.SchedulerConfig()
    # batch size
    scheduler_config.max_num_batched_tokens = 32
    # cache params
    scheduler_config.num_kv_blocks = 364
    scheduler_config.block_size = 32
    # mode - vLLM or dynamic_split_fuse
    scheduler_config.dynamic_split_fuse = True
    # vLLM specific params
    scheduler_config.max_num_seqs = 2
    scheduler_config.enable_prefix_caching = False

    draft_model = openvino_genai.DraftModel(args.draft_model_dir, device)

    ov_config = { "scheduler_config": scheduler_config, "draft_model": draft_model }

    pipe = openvino_genai.LLMPipeline(args.model_dir, device, ov_config)
    
    text_print_streamer = IterableStreamer(pipe.get_tokenizer())
    def token_printer():
        # Getting next elements from iterable will be blocked until a new token is available.
        for word in text_print_streamer:
            print(word, end='', flush=True)
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

if '__main__' == __name__:
    main()
