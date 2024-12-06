#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import threading
import queue
import openvino_genai as ov_genai


class GenaiChunkStreamer(ov_genai.StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
        tokens_cache (list): A buffer to accumulate tokens for detokenization.
        text_queue (Queue): A synchronized queue for storing decoded text chunks.
        print_len (int): The length of the printed text to manage incremental decoding.
    """

    def __init__(self, tokenizer, tokens_len=1):
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
        self.tokens_len = tokens_len

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
        if len(self.tokens_cache) % self.tokens_len == 0:
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
        else:
            return False

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
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = ov_genai.LLMPipeline(args.model_dir, device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100
    tokens_len = 10  # chunk size

    text_print_streamer = GenaiChunkStreamer(pipe.get_tokenizer(), tokens_len)

    def token_printer():
        # Getting next elements from iterable will be blocked until a new token is available.
        for word in text_print_streamer:
            print(word, end='', flush=True)

    pipe.start_chat()
    while True:
        try:
            prompt = input('question:\n')
        except EOFError:
            break
        printer_thread = threading.Thread(target=token_printer, daemon=True)
        printer_thread.start()
        pipe.generate(prompt, config, streamer=text_print_streamer)
        printer_thread.join()
        print('\n----------')
    pipe.finish_chat()


if '__main__' == __name__:
    main()
