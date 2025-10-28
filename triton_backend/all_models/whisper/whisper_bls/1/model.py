# # -*- coding: utf-8 -*-
# import json
# import re
# import traceback

# import numpy as np
# import torch
# import triton_python_backend_utils as pb_utils
# from torch.utils.dlpack import to_dlpack

# from .fbank import FeatureExtractor
# from .tokenizer import get_tokenizer


# class TritonPythonModel:
#     """Your Python model must use the same class name. Every Python model
#     that is created must have "TritonPythonModel" as the class name.
#     """

#     def initialize(self, args):
#         """`initialize` is called only once when the model is being loaded.
#         Implementing `initialize` function is optional. This function allows
#         the model to initialize any state associated with this model.

#         Parameters
#         ----------
#         args : dict
#           Both keys and values are strings. The dictionary keys and values are:
#           * model_config: A JSON string containing the model configuration
#           * model_instance_kind: A string containing model instance kind
#           * model_instance_device_id: A string containing model instance device ID
#           * model_repository: Model repository path
#           * model_version: Model version
#           * model_name: Model name
#         """
#         self.model_config = json.loads(args['model_config'])

#         self.tokenizer = get_tokenizer(num_languages=100)
#         self.eos = self.tokenizer.encode(
#             "<|endoftext|>",
#             allowed_special=self.tokenizer.special_tokens_set)[0]
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
#             self.model_config)
#         self.logger = pb_utils.Logger
#         self.init_model(self.model_config['parameters'])

#     def init_model(self, parameters):
#         for key, value in parameters.items():
#             parameters[key] = value["string_value"]
#         n_mels = int(parameters["n_mels"])
#         self.zero_pad = True if parameters["zero_pad"] == "true" else False
#         self.feature_extractor = FeatureExtractor(n_mels=n_mels)

#     def _prepare_inputs(self,
#                         request,
#                         mel_feature,
#                         mel_len,
#                         prompt,
#                         max_tokens=50):
#         """
#         Prepares inputs for the language model based on the parameters in the
#         request, image features, and prompt. It tokenizes prompt,
#         extracts and processes additional parameters from the request:
#             - max_tokens: Maximum number of tokens to generate (default: 50)
#             - temperature: Controls randomness in generation (default: 0.5)
#             - top_k: Top K sampling parameter (default: 1)
#             - frequency_penalty: Penalizes frequent tokens (default: 0.7)
#             - seed: Random seed for generation (default: 10)

#         Final llm input dictionary is combined out of all processed parameters,
#         prompt's tokens and image features. The latter will be passed to llm
#         through `prompt_embedding_table`.

#         Parameters
#         ----------
#         - request: The original request object containing additional parameters.
#         - image_features (list): A list containing image feature tensors.
#         - prompt (str): The text prompt to be processed.

#         Returns
#         -------
#         - dict: A dictionary containing all the prepared inputs for the language model.
#         """
#         input_dict = {
#             "request_output_len": np.array([[max_tokens]], dtype=np.int32),
#             "end_id": np.array([[self.eos]], dtype=np.int32),
#             "pad_id": np.array([[self.eos]], dtype=np.int32),
#             "encoder_output_lengths": mel_len // 2,
#             "input_lengths": mel_len,
#             "decoder_input_ids": prompt,
#             "streaming": np.array([[self.decoupled]], dtype=np.bool_),
#             # Ask TRT-LLM to return token-level log-probabilities
#             "return_log_probs": np.array([[True]], dtype=np.bool_),
#         }
#         input_tensor_list = [
#             pb_utils.Tensor(k, v) for k, v in input_dict.items()
#         ]
#         input_tensor_list.append(
#             pb_utils.Tensor.from_dlpack("encoder_input_features",
#                                         to_dlpack(mel_feature.contiguous())))

#         return input_tensor_list

#     def _prepare_llm_response(self, llm_request_inputs):
#         """
#         Prepares the response from the language model based on the provided
#         inputs. Creates a `pb_utils.InferenceRequest` object with passed
#         `llm_request_inputs` to send to a decoupled TensorRTLLM model.
#         For each response from the language model:
#             - Checks for errors and raise an exception if any are found.
#             - Extracts the "output_ids" tensor from the response.
#             - Determines the finish reason based on the presence of the
#               end-of-sequence token or reaching the maximum length.
#             - Appends the generated token IDs to `output_ids`.
#             - If the finish reason is determined, decodes the output IDs to text
#               and prepares the final response.

#         The final response includes the generated text, finish reason,
#         completion tokens, prompt tokens, and total tokens.

#         Parameters
#         ----------
#         - llm_request_inputs (dict): A dictionary containing the inputs for the language model.

#         Returns
#         -------
#         - pb_utils.InferenceResponse: The response object containing the generated text and additional metadata.
#         """
#         llm_request = pb_utils.InferenceRequest(
#             model_name="tensorrt_llm",
#             requested_output_names=["output_ids", "sequence_length", "output_log_probs", "cum_log_probs"],
#             inputs=llm_request_inputs,
#         )
#         responses = llm_request.exec(decoupled=self.decoupled)
#         if not self.decoupled:
#             llm_response = responses
#             if llm_response.has_error():
#                 raise pb_utils.TritonModelException(
#                     llm_response.error().message())
            
#             output_token_ids = (pb_utils.get_output_tensor_by_name(
#                 llm_response, "output_ids").as_numpy().flatten().tolist())
#             output_log_probs = pb_utils.get_output_tensor_by_name(
#                 llm_response, "output_log_probs").as_numpy()
#             cum_log_probs = pb_utils.get_output_tensor_by_name(
#                 llm_response, "cum_log_probs").as_numpy()

#             output_text = self.tokenizer.decode(output_token_ids).strip()
#             output_text = re.sub(r'<\|.*?\|>', '', output_text)

#             # Ensure correct shapes and data types
#             output_token_ids_array = np.array(output_token_ids, dtype=np.int32)
#             output_log_probs_flat = output_log_probs.flatten().astype(np.float32)
#             cum_log_probs_flat = cum_log_probs.flatten().astype(np.float32)

#             output_tensors = [
#                 pb_utils.Tensor("TRANSCRIPTS", np.array([output_text], dtype=np.object_)),
#                 pb_utils.Tensor("OUTPUT_TOKEN_IDS", output_token_ids_array),
#                 pb_utils.Tensor("CUM_LOG_PROBS", np.expand_dims(cum_log_probs_flat, 0)),
#                 pb_utils.Tensor("OUTPUT_LOG_PROBS", np.expand_dims(output_log_probs_flat, 0)), 
#             ]
#             response = pb_utils.InferenceResponse(output_tensors)
#             yield response
#         else:
#             output_token_ids = []
#             output_log_probs_list = []
#             cum_log_probs_list = []

#             for llm_response in responses:
#                 if llm_response.has_error():
#                     raise pb_utils.TritonModelException(
#                         llm_response.error().message())
#                 stream_output_ids = (pb_utils.get_output_tensor_by_name(
#                     llm_response, "output_ids").as_numpy().flatten().tolist())
#                 if len(stream_output_ids) == 0:
#                     continue
#                 output_token_ids.extend(stream_output_ids)

#                 stream_log_probs = pb_utils.get_output_tensor_by_name(
#                     llm_response, "output_log_probs").as_numpy().flatten().tolist()
#                 output_log_probs_list.extend(stream_log_probs)

#                 # Try to get cum_log_probs if available in streaming mode
#                 try:
#                     stream_cum_log_probs = pb_utils.get_output_tensor_by_name(
#                         llm_response, "cum_log_probs")
#                     if stream_cum_log_probs is not None:
#                         cum_log_probs_list.append(stream_cum_log_probs.as_numpy())
#                 except:
#                     pass

#                 output_text = self.tokenizer.decode(output_token_ids).strip()
#                 output_text = re.sub(r'<\|.*?\|>', '', output_text)

#                 # Ensure correct data types
#                 output_token_ids_array = np.array(output_token_ids, dtype=np.int32)
#                 output_log_probs_array = np.array(output_log_probs_list, dtype=np.float32)

#                 output_tensors = [
#                     pb_utils.Tensor("TRANSCRIPTS", np.array([output_text], dtype=np.object_)),
#                     pb_utils.Tensor("OUTPUT_TOKEN_IDS", output_token_ids_array),
#                     pb_utils.Tensor("OUTPUT_LOG_PROBS", np.expand_dims(output_log_probs_array, 0)),
#                 ]

#                 # Add CUM_LOG_PROBS only if available
#                 if cum_log_probs_list:
#                     cum_log_probs_array = np.concatenate(cum_log_probs_list, axis=0).astype(np.float32)
#                     output_tensors.append(
#                         pb_utils.Tensor("CUM_LOG_PROBS", np.expand_dims(cum_log_probs_array.flatten(), 0))
#                     )

#                 response = pb_utils.InferenceResponse(output_tensors=output_tensors)
#                 yield response

#     def execute(self, requests):

#         responses = []

#         for request in requests:
#             # Perform inference on the request and append it to responses list...
#             decoder_text_prompt = pb_utils.get_input_tensor_by_name(
#                 request, "TEXT_PREFIX").as_numpy().tolist()
#             text_prefix = decoder_text_prompt[0][0].decode('utf-8')
#             if text_prefix == "":
#                 text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
#             prompt_id = self.tokenizer.encode(
#                 text_prefix, allowed_special=self.tokenizer.special_tokens_set)
#             decoder_input_ids = np.array([prompt_id], dtype=np.int32)

#             wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
#             assert wav.shape[0] == 1, "Only support batch size 1"
#             # To support batch > 1
#             # cat mel,text_prompt, also, need to increase decoder_input_len as a triton input
#             wav = torch.from_numpy(wav[0]).to(self.device)
#             wav_len = pb_utils.get_input_tensor_by_name(
#                 request, "WAV_LENS").as_numpy().item()
#             if self.zero_pad:
#                 wav = wav[:wav_len]
#                 target = 0
#             else:
#                 target = 3000
            
#             mel = self.feature_extractor.compute_feature(wav, target).transpose(
#                 1, 2)
#             mel_len = np.array([[mel.shape[1]]], dtype=np.int32)
#             if self.decoupled:
#                 response_sender = request.get_response_sender()
#             try:
#                 llm_request_inputs = self._prepare_inputs(
#                     request, mel, mel_len, decoder_input_ids)
#                 if isinstance(llm_request_inputs, pb_utils.TritonError):
#                     error = pb_utils.InferenceResponse(error=llm_request_inputs)
#                     if self.decoupled:
#                         response_sender.send(
#                             error,
#                             flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
#                     else:
#                         responses.append(error)
#                 llm_responses = self._prepare_llm_response(llm_request_inputs)

#                 for triton_response in llm_responses:
#                     if self.decoupled:
#                         response_sender.send(triton_response)
#                     else:
#                         responses.append(triton_response)

#                 if self.decoupled:
#                     response_sender.send(
#                         flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

#             except Exception:
#                 self.logger.log_error(traceback.format_exc())
#                 # If encountering an error, send a response with err msg
#                 error_response = pb_utils.InferenceResponse(
#                     output_tensors=[],
#                     error=pb_utils.TritonError(traceback.format_exc()))

#                 if self.decoupled:
#                     response_sender.send(error_response)
#                     response_sender.send(
#                         flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
#                 else:
#                     responses.append(error_response)

#         if self.decoupled:
#             return None
#         else:
#             assert len(responses) == len(requests)
#             return responses

# -*- coding: utf-8 -*-
import json
import re
import traceback
from typing import List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack

from .fbank import FeatureExtractor
from .tokenizer import get_tokenizer


class TritonPythonModel:
    """Python model used by Triton. Class name must be TritonPythonModel."""

    def initialize(self, args):
        """
        Called once when the model is loaded.

        args: dict of strings (see Triton docs).
        """
        self.model_config = json.loads(args["model_config"])

        # Tokenizer and eos id
        self.tokenizer = get_tokenizer(num_languages=100)
        # tokenizer.encode returns a list; pick first id for eos special token
        self.eos = self.tokenizer.encode(
            "<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set
        )[0]

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # whether model is decoupled (streaming)
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )

        # Logger: use pb_utils.Logger if available, else a simple fallback using print
        try:
            # pb_utils.Logger takes a model name string in some Triton versions, but
            # to avoid version mismatch we'll wrap it.
            self._logger = pb_utils.Logger
            # provide wrapper functions for compatibility
            self._log_error = lambda msg: (
                self._logger.log_error(msg)
                if hasattr(self._logger, "log_error")
                else print("[ERROR]", msg)
            )
            self._log_warn = lambda msg: (
                self._logger.log_warn(msg)
                if hasattr(self._logger, "log_warn")
                else print("[WARN]", msg)
            )
            self._log_info = lambda msg: (
                self._logger.log_info(msg)
                if hasattr(self._logger, "log_info")
                else print("[INFO]", msg)
            )
        except Exception:
            # fallback
            self._logger = None
            self._log_error = lambda msg: print("[ERROR]", msg)
            self._log_warn = lambda msg: print("[WARN]", msg)
            self._log_info = lambda msg: print("[INFO]", msg)

        # Initialize other model components
        # Expect model_config['parameters'] present and formatted as Triton sets them
        self.init_model(self.model_config.get("parameters", {}))

    def init_model(self, parameters):
        """
        Convert Triton parameter dict to simple string values, and init feature extractor.
        Triton parameters look like: {'n_mels': {'string_value': '80'}, ...}
        """
        # Normalize parameters to simple dict of strings
        normalized = {}
        for key, value in parameters.items():
            # In Triton value is a dict with 'string_value' as key
            if isinstance(value, dict) and "string_value" in value:
                normalized[key] = value["string_value"]
            else:
                normalized[key] = value

        n_mels = int(normalized.get("n_mels", 80))
        self.zero_pad = True if normalized.get("zero_pad", "false").lower() == "true" else False
        self.feature_extractor = FeatureExtractor(n_mels=n_mels)

    def _prepare_inputs(
        self,
        request,
        mel_feature: torch.Tensor,
        mel_len: np.ndarray,
        prompt_ids: np.ndarray,
        max_tokens: int = 50,
    ) -> List[pb_utils.Tensor]:
        """
        Prepare the list of Triton input tensors for the tensorrt_llm model.

        Parameters
        ----------
        - request: Triton request (unused but kept for interface parity)
        - mel_feature: torch.Tensor or numpy array containing encoder features (B, C, T)
        - mel_len: numpy array shape (1,1) containing length (int32)
        - prompt_ids: numpy array with decoder_input_ids (shape (1, seq))
        - max_tokens: maximum tokens to generate
        """
        # Ensure mel_feature is a torch tensor
        if not torch.is_tensor(mel_feature):
            mel_feature = torch.from_numpy(np.asarray(mel_feature))

        if not mel_feature.is_contiguous():
            mel_feature = mel_feature.contiguous()

        # Ensure mel_len is int32 array and scalar shape (1,1)
        mel_len_arr = np.asarray(mel_len, dtype=np.int32)
        # encoder_output_lengths often expects shape (1,1) as well
        encoder_output_lengths = (mel_len_arr // 2).astype(np.int32)

        input_dict = {
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.eos]], dtype=np.int32),
            "pad_id": np.array([[self.eos]], dtype=np.int32),
            "encoder_output_lengths": encoder_output_lengths,
            "input_lengths": mel_len_arr.astype(np.int32),
            "decoder_input_ids": prompt_ids.astype(np.int32),
            "streaming": np.array([[bool(self.decoupled)]], dtype=np.bool_),
            # Ask TRT-LLM to return token-level log-probabilities
            "return_log_probs": np.array([[True]], dtype=np.bool_),
        }

        input_tensor_list = [pb_utils.Tensor(name, val) for name, val in input_dict.items()]

        # Convert mel_feature to dlpack for Triton
        try:
            dlpack = to_dlpack(mel_feature.to("cpu"))
            input_tensor_list.append(pb_utils.Tensor.from_dlpack("encoder_input_features", dlpack))
        except Exception as e:
            # If dlpack conversion fails, raise a TritonModelException
            raise pb_utils.TritonModelException(f"Failed to convert mel_feature to dlpack: {e}")

        return input_tensor_list

    def _prepare_llm_response(self, llm_request_inputs: List[pb_utils.Tensor]):
        """
        Sends an InferenceRequest to the tensorrt_llm model and yields InferenceResponse(s).

        Yields either a single response (non-decoupled) or multiple incremental responses
        (decoupled/streaming).
        """
        # Compute prompt length by finding decoder_input_ids in inputs
        prompt_length = 0
        try:
            for t in llm_request_inputs:
                if t.name() == "decoder_input_ids":
                    arr = t.as_numpy()
                    # assume shape (1, seq)
                    prompt_length = int(arr.shape[1]) if arr.ndim >= 2 else int(arr.size)
                    break
        except Exception:
            prompt_length = 0

        # Create the Triton inference request
        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length", "output_log_probs", "cum_log_probs"],
            inputs=llm_request_inputs,
        )

        responses = llm_request.exec(decoupled=self.decoupled)

        # NON-DECOUPLED (BATCH) MODE: responses is a single InferenceResponse-like object
        if not self.decoupled:
            llm_response = responses
            if llm_response.has_error():
                raise pb_utils.TritonModelException(llm_response.error().message())

            # Read outputs
            output_ids_t = pb_utils.get_output_tensor_by_name(llm_response, "output_ids")
            output_log_probs_t = pb_utils.get_output_tensor_by_name(llm_response, "output_log_probs")
            cum_log_probs_t = pb_utils.get_output_tensor_by_name(llm_response, "cum_log_probs")

            output_token_ids = output_ids_t.as_numpy().flatten().tolist() if output_ids_t is not None else []
            output_log_probs = output_log_probs_t.as_numpy().flatten() if output_log_probs_t is not None else np.array([], dtype=np.float32)
            cum_log_probs = cum_log_probs_t.as_numpy().flatten() if cum_log_probs_t is not None else np.array([], dtype=np.float32)

            # Skip prompt tokens if present
            generated_token_ids = output_token_ids[prompt_length:] if prompt_length > 0 else output_token_ids

            # Try to align log_probs with generated tokens: prefer last N entries
            if output_log_probs.size >= len(generated_token_ids) and len(generated_token_ids) > 0:
                generated_log_probs = output_log_probs[-len(generated_token_ids):].astype(np.float32)
            else:
                generated_log_probs = output_log_probs.astype(np.float32)

            # Decode generated tokens safely
            try:
                output_text = self.tokenizer.decode(generated_token_ids).strip() if len(generated_token_ids) > 0 else ""
            except Exception:
                # Fallback: join token ids as string
                output_text = ""
            output_text = re.sub(r"<\|.*?\|>", "", output_text)

            # Convert arrays for Triton outputs
            output_token_ids_array = np.array(generated_token_ids, dtype=np.int32)
            output_log_probs_array = np.array(generated_log_probs, dtype=np.float32)
            cum_log_probs_array = np.array(cum_log_probs, dtype=np.float32)

            # Ensure shapes are (1, N)
            def make_batch_dim(arr):
                if arr.ndim == 1:
                    return np.expand_dims(arr, 0)
                return arr

            output_tensors = [
                pb_utils.Tensor("TRANSCRIPTS", np.array([output_text], dtype=np.object_)),
                pb_utils.Tensor("OUTPUT_TOKEN_IDS", output_token_ids_array),
                pb_utils.Tensor("OUTPUT_LOG_PROBS", make_batch_dim(output_log_probs_array)),
                pb_utils.Tensor("CUM_LOG_PROBS", make_batch_dim(cum_log_probs_array)),
            ]

            yield pb_utils.InferenceResponse(output_tensors)

        # DECOUPLED (STREAMING) MODE: responses is an iterator/generator of partial responses
        else:
            output_token_ids = []
            output_log_probs_list = []
            cum_log_probs_list = []
            first_response = True

            for llm_response in responses:
                if llm_response.has_error():
                    raise pb_utils.TritonModelException(llm_response.error().message())

                out_ids_t = pb_utils.get_output_tensor_by_name(llm_response, "output_ids")
                if out_ids_t is None:
                    continue

                stream_output_ids = out_ids_t.as_numpy().flatten().tolist()
                if len(stream_output_ids) == 0:
                    continue

                # For the very first chunk, skip prompt tokens
                if first_response and prompt_length > 0:
                    if len(stream_output_ids) <= prompt_length:
                        # Entire chunk contains only prompt tokens: skip it and continue
                        first_response = False
                        continue
                    else:
                        stream_output_ids = stream_output_ids[prompt_length:]
                        first_response = False

                output_token_ids.extend(stream_output_ids)

                # output_log_probs
                out_logp_t = pb_utils.get_output_tensor_by_name(llm_response, "output_log_probs")
                if out_logp_t is not None:
                    try:
                        stream_log_probs = out_logp_t.as_numpy().flatten().tolist()
                        output_log_probs_list.extend(stream_log_probs)
                    except Exception:
                        # ignore probs if malformed
                        pass

                # cum_log_probs (some streaming implementations emit chunked cum_log_probs)
                try:
                    out_cum_t = pb_utils.get_output_tensor_by_name(llm_response, "cum_log_probs")
                    if out_cum_t is not None:
                        # Each cum log probs chunk may be shape (N,) or (1,N)
                        arr = out_cum_t.as_numpy()
                        if arr.ndim == 1:
                            arr = np.expand_dims(arr, 0)
                        cum_log_probs_list.append(arr.astype(np.float32))
                except Exception:
                    pass

                # Decode progressively
                try:
                    output_text = self.tokenizer.decode(output_token_ids).strip() if len(output_token_ids) > 0 else ""
                except Exception:
                    output_text = ""
                output_text = re.sub(r"<\|.*?\|>", "", output_text)

                output_token_ids_array = np.array(output_token_ids, dtype=np.int32)
                output_log_probs_array = np.array(output_log_probs_list, dtype=np.float32)

                # Build output tensors
                output_tensors = [
                    pb_utils.Tensor("TRANSCRIPTS", np.array([output_text], dtype=np.object_)),
                    pb_utils.Tensor("OUTPUT_TOKEN_IDS", output_token_ids_array),
                    pb_utils.Tensor("OUTPUT_LOG_PROBS", np.expand_dims(output_log_probs_array, 0)),
                ]

                # If we have cum_log_probs, concatenate them along time axis
                if cum_log_probs_list:
                    try:
                        cum_concat = np.concatenate(cum_log_probs_list, axis=1) if all(
                            (arr.ndim == 2 and arr.shape[0] == 1) for arr in cum_log_probs_list
                        ) else np.concatenate([arr.flatten() for arr in cum_log_probs_list], axis=0)
                        cum_concat = np.asarray(cum_concat).astype(np.float32)
                        # Ensure batch dim
                        if cum_concat.ndim == 1:
                            cum_concat = np.expand_dims(cum_concat, 0)
                        output_tensors.append(pb_utils.Tensor("CUM_LOG_PROBS", cum_concat))
                    except Exception:
                        # If concat fails, skip adding cum_log_probs for this chunk
                        pass

                yield pb_utils.InferenceResponse(output_tensors=output_tensors)

    def execute(self, requests):
        """
        Main entrypoint for Triton. Takes a list of requests and returns responses list
        for non-decoupled mode. For decoupled (streaming) mode it will use response_sender.
        """
        responses = []
        for request in requests:
            # Default response_sender if streaming
            response_sender = None
            if self.decoupled:
                response_sender = request.get_response_sender()

            try:
                # Read and prepare prompt
                decoder_text_prompt = pb_utils.get_input_tensor_by_name(request, "TEXT_PREFIX").as_numpy().tolist()
                text_prefix = decoder_text_prompt[0][0].decode("utf-8") if decoder_text_prompt else ""
                if text_prefix == "":
                    text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

                prompt_id = self.tokenizer.encode(text_prefix, allowed_special=self.tokenizer.special_tokens_set)
                decoder_input_ids = np.array([prompt_id], dtype=np.int32)

                # Read wav and lengths
                wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
                assert wav.shape[0] == 1, "Only support batch size 1"

                wav_tensor = torch.from_numpy(wav[0]).to(self.device)
                wav_len = int(pb_utils.get_input_tensor_by_name(request, "WAV_LENS").as_numpy().item())

                if self.zero_pad:
                    wav_tensor = wav_tensor[:wav_len]
                    target = 0
                else:
                    target = 3000

                # Feature extractor: compute_feature should return numpy or torch tensor
                mel = self.feature_extractor.compute_feature(wav_tensor, target)
                # Ensure shape is (B, C, T) or (C, T) depending on implementation
                # Original code transposed (1,2) — keep similar semantics: transpose if needed
                # If mel is torch tensor, ensure CPU and contiguous
                if torch.is_tensor(mel):
                    mel = mel.to("cpu").contiguous()
                else:
                    mel = np.asarray(mel)

                # The original code transposes axes and expected mel.shape[1] to be time dimension:
                # Keep behavior: if mel.ndim == 3 and shape (1, C, T) -> mel[0], else if (C, T) keep as-is.
                if isinstance(mel, np.ndarray):
                    if mel.ndim == 3:
                        # shape (B, C, T) -> we want (C, T) or keep as (1, C, T)
                        # We'll keep shape (1, C, T)
                        pass
                    elif mel.ndim == 2:
                        # (C, T) -> convert to (1, C, T)
                        mel = np.expand_dims(mel, 0)
                    # transpose to match expected (B, C, T) -> (B, T, C) if original code needed that
                    # The original code did transpose(1,2) on a numpy array that probably was (C, T) -> (T, C)
                    # To be safe, keep mel as (1, C, T) and let model decide.
                else:
                    # torch tensor
                    if mel.dim() == 2:
                        mel = mel.unsqueeze(0)  # (1, C, T)
                    # keep as is if dim==3

                # Compute mel_len: number of time steps (T)
                if torch.is_tensor(mel):
                    time_steps = mel.shape[2] if mel.dim() == 3 else (mel.shape[1] if mel.dim() == 2 else 0)
                else:
                    time_steps = mel.shape[2] if mel.ndim == 3 else (mel.shape[1] if mel.ndim == 2 else 0)

                mel_len = np.array([[int(time_steps)]], dtype=np.int32)

                # Prepare inputs
                llm_request_inputs = self._prepare_inputs(request, mel, mel_len, decoder_input_ids)

                # Get iterator/generator of responses
                llm_responses = self._prepare_llm_response(llm_request_inputs)

                # Send or collect responses
                for triton_response in llm_responses:
                    if self.decoupled:
                        # streaming mode: send immediately
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                # Signal final in decoupled mode
                if self.decoupled:
                    response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

            except Exception as e:
                # log the traceback
                tb = traceback.format_exc()
                self._log_error(tb)

                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(tb),
                )

                if self.decoupled:
                    if response_sender is not None:
                        response_sender.send(error_response)
                        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    responses.append(error_response)

        if self.decoupled:
            # decoupled models must return None
            return None
        else:
            # non-decoupled: return list of responses (must match number of requests)
            assert len(responses) == len(requests)
            return responses