# Java
Java projects

### image-compression: comparing frequency space representations
  - **Objective:** implement side by side comparison of Discrete Cosine Transform (DCT) and Discrete Wavelet Transform (DWT) algorithms for compressing raw RGB images
  - **Method:** Read an RGB file and convert the file to an 8x8 block based DCT representation (as used in the JPEG implementation) and a DWT representation (as used in the JPEG2000 implementation).
    - **Encode**
      - DCT: break up the image into 8x8 contiguous blocks of 64 pixels each and then perform a DCT for each block for each channel
      - DWT: convert each row (for each channel) into low pass and high pass coefficients followed by the same for each column applied to the output of the row processing. Recurse through the process through rows first then the columns next at each recursive iteration, each time operating on the low pass section
    - **Decode** by zeroing out the unrequested coefficients (just setting the coefficients to zero) and then perform an IDCT or an IDWT
  - **Execution:**
      1. compiling java file: `javac ImageCompression.java`
      2. executing command: `java ImageCompression <path/to/image/file.rgb> <int: number of coefficients | -1>`
        - use `-1` to create an animation showing a progressive analysis of DCT vs DWT 
      
      
  - Example:  `ImageCompression Image.rgb 16384`
  - Here you are making use of  1/16th of the total number of coefficients for decoding (16384). While the __number of coefficients are the same__ for both DCT and DWT decoding, the __exact coefficient indexes are different__
  - notes:
    - all input images are of size 512x512 (intentionally square and a power of 2 to facilitate easy encoding and decoding)
    - the algorithms, whether encoding or decoding, should work on each channel independently.
