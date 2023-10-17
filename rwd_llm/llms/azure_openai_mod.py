from langchain.llms import AzureOpenAI, type_to_cls_dict

NEW_AZURE_OPENAI_TYPE = "new_azure_openai"


class NewAzureOpenAI(AzureOpenAI):
    @property
    def _invocation_params(self):
        params = super()._invocation_params
        # fix InvalidRequestError: logprobs, best_of and echo parameters are not
        # available on gpt-35-turbo model.
        unsupported_params_and_defaults = {
            "logprobs": None,
            "best_of": 1,
            "echo": False,
        }
        if self.deployment_name == "gpt-35-turbo":
            for param, default in unsupported_params_and_defaults.items():
                if params.get(param, default) is not default:
                    raise ValueError(f"{param} is not available on gpt-35-turbo model.")
                params.pop(param, None)
        else:
            # 'best_of' conflicts with 'n', so remove it if it's set to the default
            # value
            if "best_of" in params and params["best_of"] == 1:
                params.pop("best_of")
        return params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return NEW_AZURE_OPENAI_TYPE


# hack to register the new class
type_to_cls_dict[NEW_AZURE_OPENAI_TYPE] = NewAzureOpenAI
