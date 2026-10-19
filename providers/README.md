# External image-generation provider

Put an executable Python adapter or executable here, then configure
FRONTIER_GENERATOR_COMMAND to point to it. No image API is bundled.
The provider must read FRONTIER_CANVAS, FRONTIER_MASK, FRONTIER_PROMPT,
FRONTIER_CONTEXT and write a PNG to FRONTIER_OUTPUT without modifying
the fixed (black) mask area.
