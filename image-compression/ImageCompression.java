//package hw2;

import static java.lang.Math.cos;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import javax.imageio.ImageIO;
import java.awt.Graphics;



import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingConstants;
import java.awt.Component;


public class ImageCompressionSaveOutput{
    JFrame frame = new JFrame();
    GridBagLayout gLayout = new GridBagLayout();
    JLabel lbC = new JLabel();
    JLabel lbW = new JLabel();
    JLabel lbCimg = new JLabel();
    JLabel lbWimg = new JLabel();
    int W = 512;
    int H = 512;
    static String inputFileStr;
    static String outdirpath;
    BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
    BufferedImage dctImg = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);;
    BufferedImage dwtImg = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);;

    static double[][] cosFilter = new double[8][8];
    //Matrix for original R, G, B
    int[][] rMatrix = new int[H][W];
    int[][] gMatrix = new int[H][W];
    int[][] bMatrix = new int[H][W];

    //Matrix for DCT of R, G, B
    int[][] rMatrix_DCT = new int[H][W];
    int[][] gMatrix_DCT = new int[H][W];
    int[][] bMatrix_DCT = new int[H][W];

    //Matrix for IDCT of R, G, B
    int[][] rMatrix_IDCT = new int[H][W];
    int[][] gMatrix_IDCT = new int[H][W];
    int[][] bMatrix_IDCT = new int[H][W];

    //Matrix for DWT of R, G, B
    double[][] rMatrix_DWT = new double[H][W];
    double[][] gMatrix_DWT = new double[H][W];
    double[][] bMatrix_DWT = new double[H][W];

    //Matrix for IDWT of R, G, B
    int[][] rMatrix_IDWT = new int[H][W];
    int[][] gMatrix_IDWT = new int[H][W];
    int[][] bMatrix_IDWT = new int[H][W];

    public static BufferedImage getScreenShot(Component component) {
        
        BufferedImage image = new BufferedImage(component.getWidth(), component.getHeight(), BufferedImage.TYPE_INT_RGB);
        // paints into image's Graphics
        component.paint(image.getGraphics());
        return image;
    }

    public static void getSaveSnapShot(Component component, String fileName) {
        try{
        //System.out.println(inputFileStr);
        BufferedImage img = new BufferedImage(component.getWidth(), component.getHeight(), BufferedImage.TYPE_INT_RGB);
        component.print(img.getGraphics()); // or: panel.printAll(...);

        //component.paint(img.getGraphics());


        // write the captured image as a PNG
        ImageIO.write(img, "png", new File(fileName)); } catch (IOException e) {
            e.printStackTrace();
        }
        /*BufferedImage bi = new BufferedImage(this.getSize().width, this.getSize().height, BufferedImage.TYPE_INT_ARGB); 
            Graphics g = bi.createGraphics();
            this.paint(g);  //this == JComponent
            g.dispose();
            try{ImageIO.write(bi,"png",new File("test.png"));}catch (Exception e) {}*/

    }

    /**
     * Read original image and start DCt and DWT process as per algorithms
     */
    public void processImages(String[] args) {

        try {
            File file = new File(args[0]);
            InputStream is = new FileInputStream(file);

            long len = file.length();
            byte[] bytes = new byte[(int)len];

            int offset = 0;
            int numRead = 0;
            while (offset < bytes.length && (numRead=is.read(bytes, offset, bytes.length-offset)) >= 0) {
                offset += numRead;
            }

            int ind = 0;
            for(int i = 0; i < H; i++){
                for(int j = 0; j < W; j++){
                    int r = bytes[ind];
                    int g = bytes[ind+H*W];
                    int b = bytes[ind+H*W*2];

                    //convert to unsigned int
                    r = r & 0xFF;
                    g = g & 0xFF;
                    b = b & 0xFF;

                    //Store the original R,G,B values
                    rMatrix[i][j] = r;
                    gMatrix[i][j] = g;
                    bMatrix[i][j] = b;

                    int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
                    //int pix = ((a << 24) + (r << 16) + (g << 8) + b);
                    img.setRGB(j,i,pix);
                    ind++;
                }
            }

            //Calculate the initial cosine transform values in matrix
            for(int i = 0;i<8;i++) {
                for(int j = 0;j<8;j++) {
                    cosFilter[i][j] = cos((2*i+1)*j*3.14159/16.00);
                }
            }

            int n = Integer.parseInt(args[1]);
            if(n != -1){
                int m = n/4096;
                //System.out.println("m = "+ m + " n="+n);

                //Discrete Cosine Transform (DCT): Do DCT and Zig-Zag traversal
                dctTransformQuantize(rMatrix, gMatrix, bMatrix, m);

                //Do IDCT
                inverseDCTTransform();

                //Discrete Wavelet Transform (DWT): Do DWT and Zig-Zag traversal
                rMatrix_DWT = dwtStandardDecomposition(rMatrix, n);
                gMatrix_DWT = dwtStandardDecomposition(gMatrix, n);
                bMatrix_DWT = dwtStandardDecomposition(bMatrix, n);

                rMatrix_IDWT = idwtComposition(rMatrix_DWT);
                gMatrix_IDWT = idwtComposition(gMatrix_DWT);
                bMatrix_IDWT = idwtComposition(bMatrix_DWT);

                //Display DCT and DWT Image
                displayDctDwtImage(0);

            }else{
                int iteration = 1;
                for(int i=4096; i <= 512*512; i=i+4096) {

                    n = i;
                    int m = n/4096;
                    //System.out.println("iteration="+ iteration + " n = "+ i + " m="+m);

                    //Discrete Cosine Transform (DCT): Do DCT and Zig-Zag traversal
                    dctTransformQuantize(rMatrix, gMatrix, bMatrix, m);

                    //Do IDCT
                    inverseDCTTransform();

                    //Discrete Wavelet Transform (DWT): Do DWT and Zig-Zag traversal
                    rMatrix_DWT = dwtStandardDecomposition(rMatrix, n);
                    gMatrix_DWT = dwtStandardDecomposition(gMatrix, n);
                    bMatrix_DWT = dwtStandardDecomposition(bMatrix, n);

                    rMatrix_IDWT = idwtComposition(rMatrix_DWT);
                    gMatrix_IDWT = idwtComposition(gMatrix_DWT);
                    bMatrix_IDWT = idwtComposition(bMatrix_DWT);


                    

                    displayDctDwtImage(iteration);
                    //Display DCT and DWT Image with a delay of 1 second
                    try {
                        //displayDctDwtImage(iteration);

                        Thread.sleep(1000);
                    } catch (InterruptedException e) {}
                    
                    iteration++;

                    if(i == 512*512){	//continous loop condition
                        i = 0;
                        iteration=1;
                    }
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    /**
     * Display DCT and DWT images side by side as per compression algorithm
     */
    private void displayDctDwtImage(int iteration) throws IOException {
        //Display the Image matrix
        for(int i = 0; i < H; i++) {
            for(int j = 0; j < W; j++) {
                //Display IDCT matrix
                int r = rMatrix_IDCT[i][j];
                int g = gMatrix_IDCT[i][j];
                int b = bMatrix_IDCT[i][j];

                int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
                dctImg.setRGB(j,i,pix);

                //Display IDWT matrix
                int rr = (int) rMatrix_IDWT[i][j];
                int gg = (int) gMatrix_IDWT[i][j];
                int bb = (int) bMatrix_IDWT[i][j];

                int pixx = 0xff000000 | ((rr & 0xff) << 16) | ((gg & 0xff) << 8) | (bb & 0xff);
                dwtImg.setRGB(j,i,pixx);
            }
        }

        // Use labels to display the images
        frame.getContentPane().setLayout(gLayout);

        lbC.setText(iteration != 0 ? "DCT (Iteration : "+iteration+")" : "DCT");
        lbC.setHorizontalAlignment(SwingConstants.CENTER);
        lbW.setText(iteration != 0 ? "DWT (Iteration : "+iteration+")" : "DWT");
        lbW.setHorizontalAlignment(SwingConstants.CENTER);
        lbCimg.setIcon(new ImageIcon(dctImg));
        lbWimg.setIcon(new ImageIcon(dwtImg));

        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.HORIZONTAL;
        c.anchor = GridBagConstraints.CENTER;
        c.weightx = 0.5;
        c.gridx = 0;
        c.gridy = 0;
        frame.getContentPane().add(lbC, c);

        c.fill = GridBagConstraints.HORIZONTAL;
        c.anchor = GridBagConstraints.CENTER;
        c.weightx = 0.5;
        c.gridx = 1;
        c.gridy = 0;
        frame.getContentPane().add(lbW, c);

        c.fill = GridBagConstraints.HORIZONTAL;
        c.gridx = 0;
        c.gridy = 1;
        frame.getContentPane().add(lbCimg, c);

        c.fill = GridBagConstraints.HORIZONTAL;
        c.gridx = 1;
        c.gridy = 1;
        frame.getContentPane().add(lbWimg, c);

        frame.pack();
        frame.setVisible(true);

        // capture and save image
        String outputFilename=outdirpath+Integer.toString(iteration)+".png";
        getSaveSnapShot(frame, outputFilename);
        /*try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {}
*/

        /* 
        String filePath=inputFileStr;
        System.out.println(filePath);

        String temp = filePath.split(".+?/(?=[^/]+$)")[1]; // split filepath into [dir, filename]
        String fileName=temp.split(".")[-1];
        //String outdirpath = "/Users/kaixiwang/localUSC/CSCI576/saved_outputs/"+fileName+"/";
        //File outdir = new File(outdirpath);
        outdir.mkdir();
        String outputFilename=outdirpath+Integer.toString(iteration)+".png";
        getSaveSnapShot(frame, outputFilename);

        // alternative code to capture and save image
        String filePath=inputFileStr;
        System.out.println(filePath);

        String fileName = filePath.split(".+?/(?=[^/]+$)")[1]; // split filepath into [dir, filename]
        //String fileName=temp.split(".");
        String outdirpath = "/Users/kaixiwang/localUSC/CSCI576/saved_outputs/"+fileName+"/";
        File outdir = new File(outdirpath);
        outdir.mkdir();
        String outputFilename=outdirpath+Integer.toString(iteration)+".png";
        getSaveSnapShot(frame, outputFilename);
        //BufferedImage screenshot = getScreenShot(frame);
        System.out.println(outputFilename);
        //ImageIO.write(screenshot, "png", new File(outputFilename));*/
    }

    /**
     * Create Matrix Transpose
     * @param matrix
     * @return
     */
    private double[][] transpose(double[][] matrix) {
        double[][] temp = new double[H][W];
        for(int i=0; i<H; i++) {
            for(int j=0; j< W; j++) {
                temp[i][j] = matrix[j][i];
            }
        }
        return temp;
    }

    /**
     * DWT Standard decomposition function
     * @param matrix
     * @param n
     * @return
     */
    private double[][] dwtStandardDecomposition(int[][] matrix,int n) {
        //Copy to a double matrix
        double[][] dMatrix = new double[H][W];
        for(int i=0; i<H; i++)
            for(int j=0; j<W; j++)
                dMatrix[i][j] = matrix[i][j];

        //All rows first
        for(int row=0; row < W; row++){
            dMatrix[row] = decomposition(dMatrix[row]);
        }

        //Then all columns
        dMatrix = transpose(dMatrix);
        for(int col=0; col < H; col++) {
            dMatrix[col] = decomposition(dMatrix[col]);
        }
        dMatrix = transpose(dMatrix);

        //Do Zig Zag traversal
        dMatrix = zigZagTraversal(dMatrix, n);

        return dMatrix;
    }

    private double[] decomposition(double[] array) {
        int h = array.length;
        while (h > 0) {
            array = decompositionStep(array, h);
            h = h/2;
        }
        return array;
    }

    private double[] decompositionStep(double[] array, int h) {
        double[] dArray = Arrays.copyOf(array, array.length);
        for(int i=0; i < h/2; i++) {
            dArray[i] = (array[2*i] + array[2*i + 1]) / 2; //Low pass output
            dArray[h/2 + i] = (array[2*i] - array[2*i + 1]) / 2;	//High pass output
        }
        return dArray;
    }


    /**
     * IDWT Algorithm doing reverse of DWT and recreating the image
     * @param matrix
     * @return
     */
    private int[][] idwtComposition(double[][] matrix) {
        int[][] iMatrix = new int[H][W];

        //First Columns
        matrix = transpose(matrix);
        for(int col=0; col < H; col++) {
            matrix[col] = composition (matrix[col]);
        }
        matrix = transpose(matrix);

        //Then all rows
        for(int row=0; row < W; row++){
            matrix[row] = composition(matrix[row]);
        }

        //Copy the composition matrix to int matrix of R,G,B values. Limit values 0 to 255
        for(int i=0; i < H; i++){
            for(int j=0; j<W; j++){
                iMatrix[i][j] = (int) Math.round(matrix[i][j]);
                if(iMatrix[i][j] < 0){
                    iMatrix[i][j] = 0;
                }
                if(iMatrix[i][j] > 255){
                    iMatrix[i][j] = 255;
                }
            }
        }

        return iMatrix;
    }

    private double[] composition(double[] array) {
        int h = 1;
        while (h <= array.length) {
            array = compositionStep(array, h);
            h = h*2;
        }
        return array;
    }

    private double[] compositionStep(double[] array, int h) {
        double[] dArray = Arrays.copyOf(array, array.length);
        for(int i=0; i < h/2; i++) {
            dArray[2*i] = array[i] + array[h/2 + i];
            dArray[2*i + 1] = array[i] - array[h/2 + i];
        }
        return dArray;
    }


    /**
     * DCT Transformation as per compression Algorithm
     * @param rMatrix
     * @param gMatrix
     * @param bMatrix
     * @param m
     */
    private void dctTransformQuantize(int[][] rMatrix, int[][] gMatrix, int[][] bMatrix, int m) {

        int H = 512;
        int W = 512;

        for(int i = 0; i < H; i+=8) {
            for(int j = 0; j < W;j+=8) {

                //Store block values
                double[][] rBlock = new double[8][8];
                double[][] gBlock = new double[8][8];
                double[][] bBlock = new double[8][8];

                for(int u = 0; u < 8; u++) {
                    for(int v = 0; v < 8; v++) {

                        float cu = 1.0f, cv = 1.0f;
                        float rResult = 0.00f, gResult = 0.00f, bResult = 0.00f;

                        if(u == 0)
                            cu =  0.707f;
                        if(v == 0)
                            cv = 0.707f;

                        for(int x = 0; x<8; x++) {
                            for(int y = 0;y<8;y++) {

                                int iR, iG, iB;

                                iR = (int) rMatrix[i+x][j+y];
                                iG = (int) gMatrix[i+x][j+y];
                                iB = (int) bMatrix[i+x][j+y];

                                rResult += iR*cosFilter[x][u]*cosFilter[y][v];
                                gResult += iG*cosFilter[x][u]*cosFilter[y][v];
                                bResult += iB*cosFilter[x][u]*cosFilter[y][v];

                            }
                        }//end x

                        rBlock[u][v] = (int) Math.round(rResult * 0.25*cu*cv);
                        gBlock[u][v] = (int) Math.round(gResult * 0.25*cu*cv);
                        bBlock[u][v] = (int) Math.round(bResult * 0.25*cu*cv);
                    }//end v
                }//end u

                //Do Zig Zag traversal
                rBlock = zigZagTraversal(rBlock, m);
                gBlock = zigZagTraversal(gBlock, m);
                bBlock = zigZagTraversal(bBlock, m);

                for(int u = 0; u < 8; u++) {
                    for(int v = 0; v < 8; v++) {
                        rMatrix_DCT[i+u][j+v] = (int) rBlock[u][v];
                        gMatrix_DCT[i+u][j+v] = (int) gBlock[u][v];
                        bMatrix_DCT[i+u][j+v] = (int) bBlock[u][v];
                    }//end v
                }//end u

            }
        }

    }

    /**
     * Zig-zag matrix traversal for counting the coefficients
     * @param matrix
     * @param m
     * @return
     */
    public double[][] zigZagTraversal(double[][] matrix, int m) {
        int i = 0;
        int j = 0;
        int length = matrix.length-1;
        int count = 1;

        //for upper triangle of matrix
        if(count > m){
            matrix[i][j]=0; count++;
        }else{
            count++;
        }

        while(true) {

            j++;
            if(count > m){
                matrix[i][j]=0; count++;
            }else{
                count++;
            }

            while(j!=0) {
                i++;
                j--;

                if(count > m){
                    matrix[i][j]=0; count++;
                }else{
                    count++;
                }
            }
            i++;
            if(i > length) {
                i--;
                break;
            }

            if(count > m){
                matrix[i][j]=0; count++;
            }else{
                count++;
            }

            while(i!=0) {
                i--;
                j++;
                if(count > m){
                    matrix[i][j]=0; count++;
                }else{
                    count++;
                }
            }
        }

        //for lower triangle of matrix
        while(true) {
            j++;
            if(count > m){
                matrix[i][j]=0; count++;
            }else{
                count++;
            }

            while(j != length)
            {
                j++;
                i--;

                if(count > m){
                    matrix[i][j]=0; count++;
                }else{
                    count++;
                }
            }
            i++;
            if(i > length)
            {
                i--;
                break;
            }

            if(count > m){
                matrix[i][j]=0; count++;
            }else{
                count++;
            }

            while(i != length)
            {
                i++;
                j--;
                if(count > m){
                    matrix[i][j]=0; count++;
                }else{
                    count++;
                }
            }
        }
        return matrix;
    }

    /**
     * Inverse DCT Transformation as per formula doing the computation
     */
    public void inverseDCTTransform() {
        int H = 512;
        int W = 512;

        for(int i = 0;i<H;i+=8) {
            for(int j = 0;j<W;j+=8) {
                for(int x = 0;x<8;x++) {
                    for(int y = 0;y<8;y++) {
                        float fRRes = 0.00f, fGRes = 0.00f, fBRes = 0.00f;

                        for(int u = 0;u<8;u++) {
                            for(int v = 0;v<8;v++) {
                                float fCu = 1.0f, fCv = 1.0f;
                                if(u == 0)
                                    fCu =  0.707f;
                                if(v == 0)
                                    fCv = 0.707f;

                                double iR, iG, iB;
                                iR = rMatrix_DCT[i + u][j + v];
                                iG = gMatrix_DCT[i + u][j + v];
                                iB = bMatrix_DCT[i + u][j + v];

                                //IDCT Formula calculation
                                fRRes += fCu * fCv * iR*cosFilter[x][u]*cosFilter[y][v];
                                fGRes += fCu * fCv * iG*cosFilter[x][u]*cosFilter[y][v];
                                fBRes += fCu * fCv * iB*cosFilter[x][u]*cosFilter[y][v];
                            }
                        }

                        fRRes *= 0.25;
                        fGRes *= 0.25;
                        fBRes *= 0.25;

                        //Check R, G, B values for overflow and limit it between 0 to 255
                        if(fRRes <= 0)
                            fRRes = 0;
                        else if(fRRes >= 255)
                            fRRes = 255;

                        if(fGRes <= 0)
                            fGRes = 0;
                        else if(fGRes >= 255)
                            fGRes = 255;

                        if(fBRes <= 0)
                            fBRes = 0;
                        else if(fBRes >= 255)
                            fBRes = 255;

                        rMatrix_IDCT[i + x][j + y]  = (int)fRRes;
                        gMatrix_IDCT[i + x][j + y]  = (int)fGRes;
                        bMatrix_IDCT[i + x][j + y]  = (int)fBRes;
                    }
                }
            }
        }

    }

    public static void main(String[] args) {
        //Input example:
        //D:\\workspace\\....path to file....\\rgbimages\\Lenna.rgb -1
        String fileName = args[0].split(".+?/(?=[^/]+$)")[1]; // split filepath into [dir, filename]
        outdirpath = "/Users/kaixiwang/localUSC/CSCI576/saved_outputs/"+fileName+"/";
        File outdir = new File(outdirpath);
        outdir.mkdir();
        System.out.println("outputs will be saved to: "+outdirpath);



        ImageCompressionSaveOutput ren = new ImageCompressionSaveOutput();
        ren.processImages(args);
    }
}