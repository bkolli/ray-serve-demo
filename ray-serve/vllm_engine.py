from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from starlette.responses import Response
from starlette.requests import Request
from http import HTTPStatus

from fastapi import FastAPI, HTTPException
import logging
import uuid

from ray import serve
from ray.serve import Application
from typing import Optional, Literal, List, Dict
from pydantic import BaseModel

logger = logging.getLogger("ray.serve")

app = FastAPI()

class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content

class GenerateRequest(BaseModel):
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    prompt: Optional[str]
    messages: Optional[List[Message]]

class GenerateResponse(BaseModel):
    output: Optional[str]
    finish_reason: Optional[str]
    prompt: Optional[str]

@serve.deployment(
    name="VLLMInference",
    num_replicas=1,
    max_concurrent_queries=256,
    ray_actor_options={"num_gpus": 1.0}
)
@serve.ingress(app)
class VLLMInference:
    def __init__(self, model: str, hf_auth_token: str, trust_remote_code: bool = True, dtype: str = "float16"):
        self.args = AsyncEngineArgs(
            model=model,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
        self.engine = AsyncLLMEngine.from_engine_args(self.args)
        self.tokenizer = self._prepare_tokenizer(model, hf_auth_token, trust_remote_code)

    def _prepare_tokenizer(self, model: str, hf_auth_token: str, trust_remote_code: bool):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            use_auth_token=hf_auth_token
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate_text(self, request: GenerateRequest, raw_request: Request) -> GenerateResponse:
        try:
            generation_args = request.dict(exclude={"prompt", "messages"})
            generation_args = generation_args or {"max_tokens": 500, "temperature": 0.1}

            if request.prompt:
                prompt = request.prompt
            elif request.messages:
                prompt = self.tokenizer.apply_chat_template(
                    request.messages, tokenize=False, add_generation_prompt=True
                )
            else:
                raise ValueError("Either 'prompt' or 'messages' must be provided.")

            sampling_params = SamplingParams(**generation_args)
            request_id = str(uuid.uuid1().hex)

            results_generator = self.engine.generate(prompt, sampling_params, request_id)

            final_result = None
            async for result in results_generator:
                if await raw_request.is_disconnected():
                    await self.engine.abort(request_id)
                    return GenerateResponse()
                final_result = result  # Store the last result

            if final_result:
                return GenerateResponse(
                    output=final_result.outputs[0].text,
                    finish_reason=final_result.outputs[0].finish_reason,
                    prompt=final_result.prompt
                )
            else:
                raise ValueError("No results found.")
        except ValueError as e:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            logger.error("Error in generate_text", exc_info=True)
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "Server error")

    @app.get("/health")
    async def health(self) -> Response:
        return Response(status_code=200)

def deployment_llm(args: dict) -> Application:
    return VLLMInference.bind(
        model=args["model"],
        hf_auth_token=args["hf_auth_token"],
        trust_remote_code=args.get("trust_remote_code", True),
        dtype=args.get("dtype", "float16"),
    )
