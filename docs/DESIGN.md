# Architecture of the Hybrid model

Implemented the Hybrid model using a interleaved architecture
---->Layer 1:Mamba
---->Layer 2:Deltanet
---->Layer 3:Mamba
---->Layer 4:Deltanet

# Why?
We are using the Mamba layer first because it has a input-dependent decay,so it compresses the prefix into a running state and writes a summary into the residual stream. If we place a deltanet layer first instead it would see only raw symbol embeddings with a width-4 convolution and since the cipher does not have short range patterns this would be not efficient. Placing the first mamba block writes a summary of the whole prefix into the residual stream, so deltanet's can work on a global summary instead of just local symbols. Deltanet can overwrite, if the cipher changes as the text goes an overwrite mechanism would be preferred.so the data flow is :
 1. The embeddings enter the stream.
 2. The first Mamba block writes a running summary of the prefix into the stream.
 3. The first DeltaNet block reads that summary, builds keys and values from it, and writes corrected lookups back.
 4. The second Mamba block summarizes the corrected stream.
 5. The second DeltaNet block does the final lookups.
 6. A final norm and the output layer turn each position’s stream into a plaintext character.
