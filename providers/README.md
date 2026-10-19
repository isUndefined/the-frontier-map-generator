# OpenAI Image Edit provider

The default provider is `openai_edit.py`, using GPT Image 2.5 native-size editing.
No resizing or upscaling: padding to multiples of 16, exact cropping, pixel-perfect
restoration of fixed mask regions, and strict output-dimension validation.

Configure `OPENAI_API_KEY` in an ignored `.env`; see
[Docker setup](../DOCKER_WORLD_GENERATOR.md).

An external image provider can still be specified with
`FRONTIER_GENERATOR_COMMAND`.
