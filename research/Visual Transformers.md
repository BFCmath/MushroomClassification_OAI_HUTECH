# Visual Transformers

## The Visual Transformer (VT) Framework

The VT framework reimagines how images are represented and processed. It consists of three main components:

1. Tokenizer:
   + Converts the feature map (initially extracted by convolutional layers) into a small set of semantic visual tokens (e.g., 16 tokens). Each token represents a high-level concept in the image, such as an object or part of the background.
   + Two types of tokenizers are proposed:
     + Filter-based Tokenizer: Uses convolutional filters to group pixels into semantic tokens, followed by spatial pooling. However, it may waste computation by modeling all possible concepts with fixed filters.
     + Recurrent Tokenizer: A content-aware approach where token extraction is guided by tokens from the previous layer, refining the process based on context.

1. Transformer:
   + Processes the visual tokens using self-attention, a mechanism that models relationships between tokens dynamically. Unlike convolutions or graph convolutions (which use fixed weights), transformers adapt their weights based on the input, allowing tokens to represent varying concepts depending on the image. This reduces the number of tokens needed (e.g., 16 vs. hundreds in graph-based methods).

1. Projector:
+ For tasks requiring pixel-level details (e.g., segmentation), the transformer’s output tokens are projected back onto the feature map using attention mechanisms. This step fuses high-level semantic information with spatial details from the original feature map.
