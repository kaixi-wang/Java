# Java Projects
## raw-video-player
**Topics:** video resampling, spatial and temporal aliasing effects, image aspect ratios and pixel aspect ratios
**Objective:** Given a video file as input, produce a spatially & temporally resampled output

#### Inputs: 
  - 6 parameters where:
      - 1. Name of the input video file
      - 2. floating pointing number that will be the scaling factor for width
      - 3. floating pointing number that will be the scaling factor for height
      - 4. output frame rate
      - 5. controls whether anti-aliasing should be turned on (0 = off, 1 = on)
      - 6. int that controls which analyses to implement for dealing with Pixel Aspect Ratio when resizing 
          - 0 = default implementation
          - 1 = nonlinear mapping 
          - 2 = seam carving ( See [http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html])
          - 3 = temporal content aware remapping
    - **Execution:**
      1. compiling java file: `javac VideoDisplay.java`
      2. `java VideoDisplay Video.rgb 0.5 0.5 10 1 0`
    
This should change your video file to half its size at the same frame rate but have anti aliasing turned on.

## image-compression: comparing frequency space representations
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
