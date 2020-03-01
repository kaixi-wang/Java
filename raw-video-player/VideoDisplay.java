/**
Kaixi Wang
Assignment 1 + Extra Credit
 */
import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.lang.*;
import java.util.*;
import javax.swing.*;
import javax.imageio.ImageIO;
import java.lang.Math;


public class VideoDisplay {
    static final int WIDTH = 960;
    static final int HEIGHT = 540;
    static final double aspectRatio=(double)(WIDTH)/(double)(HEIGHT);

    public static void main(String[] args) {

        // Get args from command line
        System.out.println("======================================================");
        String filePath = args[0];
        String[] fileName = filePath.split(".+?/(?=[^/]+$)"); // split filepath into [dir, filename]
        String[] videoName = fileName[1].split("_");
        System.out.println("Filename: " + fileName[1]);
        double widthScale = Double.parseDouble(args[1]);
        System.out.println("Scaling factor for width: " + widthScale);
        double heightScale = Double.parseDouble(args[2]);
        System.out.println("Scaling factor for height: " + heightScale);
        int fps = Integer.parseInt(args[3]);
        System.out.println("Frame rate: " + fps);
        int antiAliasing = Integer.parseInt(args[4]);
        if (antiAliasing == 1) {
            System.out.println("Anti-aliasing: ON");
        } else {
            System.out.println("Anti-aliasing: OFF");
        }
        int analysis = Integer.parseInt(args[5]);
        //testing seam carver
        //int testing=Integer.parseInt(args[6]);
        System.out.println("Analysis: " + analysis);

        System.out.println("======================================================");

        // create buffered image array to store every frame
        ArrayList<BufferedImage> images = new ArrayList<BufferedImage>();

        // Read bytes from video file
        try {
            System.out.println("Processing " + fileName[1] + " file contents...");
            File file = new File(args[0]);
            InputStream is = new FileInputStream(file);
            int frameNum=0;

            String outdirpath = "/Users/kaixiwang/Documents/USC/CSCI-576/HW1/"+videoName[0]+"_W"+args[1]+"-H"+args[2]+"_"+args[3]+"-"+args[4]+"-"+args[5]+"/";
            File outdir = new File(outdirpath);
            outdir.mkdir();
            // Get length of file and create byte array
            long len = file.length();
            byte[] bytes = new byte[(int) len];

            // Read all bytes from video file into byte array
            int offset = 0;
            int bytesRead = 0;
            
            while (offset < bytes.length && (bytesRead = is.read(bytes, offset, bytes.length - offset)) != -1) {
                offset += bytesRead;
            }

            // Create an image for each frame
            System.out.println("Generating images for each frame...");
            int index = 0;
            while (index + HEIGHT * WIDTH * 2 < len) {
                BufferedImage image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < HEIGHT; y++) {
                    for (int x = 0; x < WIDTH; x++) {
                        byte r = bytes[index];
                        byte g = bytes[index + HEIGHT * WIDTH];
                        byte b = bytes[index + HEIGHT * WIDTH * 2];
                        int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
                        image.setRGB(x, y, pix);
                        index++;
                    }
                }

                // Sampling and Quantization
                BufferedImage scaledImage = null;
                if (analysis == 1) {
                    if (widthScale != 1 || heightScale != 1) {
                        scaledImage = getNonlinearMapping(image, widthScale, heightScale);
                    } else {
                        scaledImage = image;
                    }
                    if (antiAliasing == 1) {
                        scaledImage = getAntiAliasing(scaledImage, 1, 1);
                    }
                }
                //Seam Carving
                else if (analysis >=2) {
                    if (widthScale != 1 || heightScale != 1) {
                        if (widthScale > 1 || heightScale > 1) {
                            Boolean shrink=false;
                            scaledImage = getSeamCarving(image, widthScale, heightScale, analysis,shrink);
                        }
                        else if (widthScale < 1 || heightScale < 1) {
                            Boolean shrink=true;
                            scaledImage = getSeamCarving(image, widthScale, heightScale, analysis,shrink);
                            }
                    } else {
                        scaledImage = image;

                    }
                }

                else {
                    if (antiAliasing == 1) {
                        scaledImage = getAntiAliasing(image, widthScale, heightScale);
                    } else if (widthScale != 1 || heightScale != 1) {
                        scaledImage = getScaledImage(image, widthScale, heightScale);
                    } else {
                        scaledImage = image;
                    }
                }
                images.add(scaledImage);
                frameNum++;
                if (frameNum>10){
                    String outputFilename=outdirpath+Integer.toString(frameNum)+".png";
                    ImageIO.write(scaledImage, "PNG", new File(outputFilename));
                }
                /*
                if (frameNum%25==0){
                    String outputFilename=outdirpath+Integer.toString(frameNum)+".png";
                    ImageIO.write(scaledImage, "PNG", new File(outputFilename));
                }
                else if (frameNum%25==1){
                    String outputFilename=outdirpath+Integer.toString(frameNum)+".png";
                    ImageIO.write(scaledImage, "PNG", new File(outputFilename));
                }
                else if (frameNum%25==2){
                    String outputFilename=outdirpath+Integer.toString(frameNum)+".png";
                    ImageIO.write(scaledImage, "PNG", new File(outputFilename));
                }*/

                index += WIDTH * HEIGHT * 2;
                //if (testing==1){
                 //   if (images.size()==20){
                  //      break;
                   // }
                //}
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Create frame and label to display video
        System.out.println("Success... processed " + images.size() + " images.");
        System.out.println("Playing video...");
        JFrame frame = new JFrame();
        JLabel label = new JLabel(new ImageIcon(images.get(0)));
        frame.getContentPane().add(label, BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);
        for (int i = 1; i < images.size(); i++) {
            label.setIcon(new ImageIcon(images.get(i)));
            try {
                Thread.sleep(1000 / fps);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        System.out.println("Finished!");
    }

    //Assume scaling factor <= 1

    public static BufferedImage getScaledImage(BufferedImage orig_img, double widthScale, double heightScale) {
        int new_width = (int) ((double) orig_img.getWidth() * widthScale);
        int new_height = (int) ((double) orig_img.getHeight() * heightScale);
        BufferedImage newImage = new BufferedImage(new_width, new_height, BufferedImage.TYPE_INT_RGB);
        for (int h = 0; h < new_height; h++) {
            int orig_height = (int) ((double) h / heightScale);
            for (int w = 0; w < new_width; w++) {
                int orig_width = (int) ((double) w / widthScale);

                int pix = orig_img.getRGB(orig_width, orig_height);
                newImage.setRGB(w, h, pix);

            }
        }
        return newImage;
    }

    // aliasing combined with getScaledImage
    public static BufferedImage getAntiAliasing(BufferedImage orig_img, double widthScale, double heightScale) {
        int new_width = (int) ((double) orig_img.getWidth() * widthScale);
        int new_height = (int) ((double) orig_img.getHeight() * heightScale);
        BufferedImage newImage = new BufferedImage(new_width, new_height, BufferedImage.TYPE_INT_RGB);
        for (int h = 0; h < new_height; h++) {
            int orig_height = (int) ((double) h / heightScale);
            for (int w = 0; w < new_width; w++) {
                int orig_width = (int) ((double) w / widthScale);

                Color newColor = getAvgFilter(orig_img, orig_width, orig_height);
                newImage.setRGB(w, h, newColor.getRGB());
            }
        }
        return newImage;
    }

    // Filter: 3x3 kernal averaging
    public static Color getAvgFilter(BufferedImage image, int w, int h) {
        int r = 0;
        int g = 0;
        int b = 0;
        int count = 0;
        for (int i = -1; i < 2; i++) {
            if (w + i < 0 || w + i >= image.getWidth())
                continue;
            for (int j = -1; j < 2; j++) {
                if (h + j < 0 || h + j >= image.getHeight())
                    continue;
                r += new Color(image.getRGB(w + i, h + j)).getRed();
                g += new Color(image.getRGB(w + i, h + j)).getGreen();
                b += new Color(image.getRGB(w + i, h + j)).getBlue();
                count++;
            }
        }
        return new Color(r / count, g / count, b / count);
    }

    //Analysis Part A: Aspect ratip

    //Analysis 1: Nonlinear Mapping
    public static BufferedImage getNonlinearMapping(BufferedImage orig_img, Double widthScale, Double heightScale) {
        int new_width = (int) ((double) orig_img.getWidth() * widthScale);
        int new_height = (int) ((double) orig_img.getHeight() * heightScale);
        double unscaled_pct = .4;
        double scaled_pct = 1.0 - unscaled_pct;

        BufferedImage new_image = new BufferedImage(new_width, new_height, BufferedImage.TYPE_INT_RGB);
        //double ratio = ((double) WIDTH / (double) HEIGHT) / ((double) new_width / (double) new_height);
        double ratio = (double) widthScale/heightScale;
        int x_center=new_width/2;
        int y_center=new_height/2;

        //horizontal stretching
        if (ratio > 1) {
            // take percentage along width
            double centerRatio = unscaled_pct * WIDTH / HEIGHT;
            int centerWidth = (int) (centerRatio * new_height);
            int segment1 = new_width / 2 - centerWidth / 2;
            int segment3 = new_width / 2 + centerWidth / 2;
            double widthScaleCenter = (double) (centerWidth / (unscaled_pct * WIDTH));
            double widthScaleSide = (double) (new_width - centerWidth) / (scaled_pct * WIDTH);

            BufferedImage centerSegment = orig_img.getSubimage((int)(scaled_pct/2.0*WIDTH), 0, (int)(unscaled_pct* WIDTH), HEIGHT);
            BufferedImage rightSegment = orig_img.getSubimage((int)(1.0 * WIDTH - (scaled_pct/2.0*WIDTH)), 0, (int)(scaled_pct/2.0*WIDTH), HEIGHT);
            //BufferedImage rightSegment = orig_img.getSubimage(4 * WIDTH / 5, 0, WIDTH / 5, HEIGHT);

            for (int y = 0; y < new_height; y++) {
                for (int x = 0; x <= segment1; x++) {
                    int xOrig = (int) ((double) x / widthScaleSide);
                    int yOrig = (int) ((double) y / heightScale);
                    int pix = orig_img.getRGB(xOrig, yOrig);
                    new_image.setRGB(x, y, pix);
                }
                for (int x = 0; x < segment3 - segment1; x++) {
                    int xOrig = (int) ((double) x / widthScaleCenter);
                    int yOrig = (int) ((double) y / heightScale);
                    int pix = centerSegment.getRGB(xOrig, yOrig);
                    new_image.setRGB(x + segment1, y, pix);
                }
                for (int x = 0; x < segment1; x++) {
                    int xOrig = (int) ((double) x / widthScaleSide);
                    int yOrig = (int) ((double) y / heightScale);
                    if (xOrig >= rightSegment.getWidth())
                        continue;
                    int pix = rightSegment.getRGB(xOrig, yOrig);
                    if (x + segment3 >= new_image.getWidth())
                        continue;
                    new_image.setRGB(x + segment3, y, pix);
                }
            }
        } else if (ratio < 1) {
            //vertical stretch
            // horizontal shrink


            double centerRatio = scaled_pct * HEIGHT / WIDTH;
            int centerHeight = (int) (centerRatio * new_width);
            //int centerHeight = (int) (.8*new_height);
            int segment1 = new_height / 2 - centerHeight / 2;
            int segment3 = new_height / 2 + centerHeight / 2;
            
            double heightScaleCenter = (double) (centerHeight / (unscaled_pct * HEIGHT));
            double heightScaleSide = (double) (new_height - centerHeight) / (scaled_pct * (double)(HEIGHT));

            BufferedImage centerSegment = orig_img.getSubimage(0, (int)(scaled_pct*.5*HEIGHT), WIDTH,  (int)(unscaled_pct * (double)(HEIGHT)));
            BufferedImage bottomSegment = orig_img.getSubimage(0, HEIGHT - (int) (scaled_pct*0.5*(double)(HEIGHT)), WIDTH, (int)(scaled_pct*0.5*(double)(HEIGHT)));

            for (int x = 0; x < new_width; x++) {
                for (int y = 0; y <= segment1; y++) {
                    int yOrig = (int) ((double) y / heightScaleSide);
                    int xOrig = (int) ((double) x / widthScale);
                    int pix = orig_img.getRGB(xOrig, yOrig);
                    new_image.setRGB(x, y, pix);
                }
                for (int y = 0; y < segment3 - segment1; y++) {
                    int yOrig = (int) ((double) y / heightScaleCenter);
                    int xOrig = (int) ((double) x / widthScale);
                    int pix = centerSegment.getRGB(xOrig, yOrig);
                    new_image.setRGB(x, y + segment1, pix);
                }
                for (int y = 0; y < segment1 - 1; y++) {
                    int yOrig = (int) ((double) y / heightScaleSide);
                    int xOrig = (int) ((double) x / widthScale);
                    if (yOrig >= bottomSegment.getHeight())
                        continue;
                    int pix = bottomSegment.getRGB(xOrig, yOrig);
                    if (y + segment3 >= new_image.getHeight())
                        continue;
                    new_image.setRGB(x, y + segment3, pix);
                }
            }
        } else {
            new_image = getScaledImage(orig_img, widthScale, heightScale);
        }
        return new_image;
    }

    public static BufferedImage getSeamCarving(BufferedImage orig_img, Double widthScale, Double heightScale, int analysis, Boolean shrink) {
        BufferedImage new_img = orig_img;
        int new_width = (int) ((double) orig_img.getWidth() * widthScale);
        int new_height = (int) ((double) orig_img.getHeight() * heightScale);
        int w = new_img.getWidth();
        int h = new_img.getHeight();

        if (analysis==2){
            if (shrink == true){
                while (w > new_width || h > new_height) {
                    if (w> new_width) {
                        new_img = carveSeam(new_img, "v");
                        w--;
                    }
                    if (h> new_height)
                    {
                        new_img = carveSeam(new_img, "h");
                        h--;
                    }
                }
            }
            if (shrink == false){

                while (w < new_width || h < new_height) {
                    if (w< new_width) {
                        new_img = carveSeam(new_img, "v");
                        w++;
                    }
                    if (h< new_height)
                    {
                        new_img = carveSeam(new_img, "h");
                        h++;
                    }
                }
            }
        }
        
        else{
            while (w > new_width || h > new_height) {
                if (w> new_width) {
                    new_img = carveSeamTemporal(new_img, "v");
                    w--;
                }
                if (h> new_height)
                {
                    new_img = carveSeamTemporal(new_img, "h");
                    h--;
                }
        }
        }
       
        //showImage(new_img);

        return new_img;
    }


    /**
     * carveSeam() takes an image and removes a single seam from that image in the
     * desired direction.
     *
     * @param image to be carved and direction of the seam (vertical / horizontal).
     * @return carved image.
     */
    private static BufferedImage carveSeam(BufferedImage image, String direction) {
        // We need to compute the energy table, find and remove a seam.
        BufferedImage newImage = null;
        double[][] energyTable = computeEnergy(image);
        int[][] seam = findSeam(energyTable, direction);
        newImage = removeSeam(image, seam, direction);

        return newImage;
    }

    //private static BufferedImage carveSeamTemporal(BufferedImage orig_img, Double widthScale, Double heightScale) {
        // We need to compute the energy table, find and remove a seam.
        //
    private static BufferedImage carveSeamTemporal(BufferedImage image, String direction) {
        BufferedImage newImage = null;
        double[][] energyTable = computeEnergyTemporal(image);
        int[][] seam = findSeam(energyTable, direction);
        newImage = removeSeam(image, seam, direction);

        return newImage;
    }


    /**
     * computeEnergy() takes an image and computes the energy table for that image.
     * The energy of a pixel is the difference in the color of the pixels next to it
     * (vertical and horizontal). If the pixel is at the edge the pixel itself replaces
     * the pixel that is 'missing'.
     *
     * @param image.
     * @return energy table (double[][]).
     */
    private static double[][] computeEnergy(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] energyTable = new double[width][height];

        // Loop over every pixel in the image and compute its energy.
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int x1Pixel;
                int x2Pixel;
                int y1Pixel;
                int y2Pixel;

                if (x == 0) {
                    // leftmost column
                    x1Pixel = image.getRGB(x, y);
                    x2Pixel = image.getRGB(x + 1, y);
                } else if (x == width - 1) {
                    // rightmost column
                    x1Pixel = image.getRGB(x - 1, y);
                    x2Pixel = image.getRGB(x, y);
                } else {
                    // middle columns
                    x1Pixel = image.getRGB(x - 1, y);
                    x2Pixel = image.getRGB(x + 1, y);
                }

                if (y == 0) {
                    //System.out.println("x, y: "+ x +" " +y);
                    // bottom row
                    y1Pixel = image.getRGB(x, y);
                    y2Pixel = image.getRGB(x, y + 1);
                } else if (y == height - 1) {
                    // top row
                    y1Pixel = image.getRGB(x, y - 1);
                    y2Pixel = image.getRGB(x, y);
                } else {
                    // middle rows
                    y1Pixel = image.getRGB(x, y - 1);
                    y2Pixel = image.getRGB(x, y + 1);
                }

                // we now have all the pixels we need, so we find the
                // differences between them.
                // By doing the bitwise operations we get at each individual
                // part of the color and can compare them. Each expression
                // should be close to 0 if the colors are similar.
                // Colors that are not similar will have a higher value.
                int xRed = Math.abs(((x1Pixel & 0x00ff0000) >> 16) - ((x2Pixel & 0x00ff0000) >> 16));
                int xGreen = Math.abs(((x1Pixel & 0x0000ff00) >> 8) - ((x2Pixel & 0x0000ff00) >> 8));
                int xBlue = Math.abs((x1Pixel & 0x000000ff) - (x2Pixel & 0x000000ff));

                int yRed = Math.abs(((y1Pixel & 0x00ff0000) >> 16) - ((y2Pixel & 0x00ff0000) >> 16));
                int yGreen = Math.abs(((y1Pixel & 0x0000ff00) >> 8) - ((y2Pixel & 0x0000ff00) >> 8));
                int yBlue = Math.abs((y1Pixel & 0x000000ff) - (y2Pixel & 0x000000ff));

                // We add up all the differences and call that our energy.
                double energy = xRed + xGreen + xBlue + yRed + yGreen + yBlue;

                energyTable[x][y] = energy;
            }
        }

        return energyTable;
    }
private static double[][] computeEnergyTemporal(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] energyTable = new double[width][height];
        int dist=2;

        // Loop over every pixel in the image and compute its energy.
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int x1Pixel;
                int x2Pixel;
                int y1Pixel;
                int y2Pixel;

                //if (x == 0) {
                if (x<dist){
                    // leftmost column
                    x1Pixel = image.getRGB(x, y);
                    x2Pixel = image.getRGB(x + dist, y);
                } else if (x >= width - dist-1) {
                    // rightmost column
                    x1Pixel = image.getRGB(x - dist, y);
                    x2Pixel = image.getRGB(x, y);
                } else {
                    // middle columns
                    x1Pixel = image.getRGB(x - dist, y);
                    x2Pixel = image.getRGB(x + dist, y);
                }

                if (y < dist) {
                    // bottom row
                    y1Pixel = image.getRGB(x, y);
                    y2Pixel = image.getRGB(x, y + dist);
                } else if (y >= height - dist-1) {
                    // top row
                    y1Pixel = image.getRGB(x, y - dist);
                    y2Pixel = image.getRGB(x, y);
                } else {
                    // middle rows
                    y1Pixel = image.getRGB(x, y - dist);
                    y2Pixel = image.getRGB(x, y + dist);
                }

                // we now have all the pixels we need, so we find the
                // differences between them.
                // By doing the bitwise operations we get at each individual
                // part of the color and can compare them. Each expression
                // should be close to 0 if the colors are similar.
                // Colors that are not similar will have a higher value.

                /*int xRed = Math.abs(((x1Pixel & 0x00ff0000) >> 16) - ((x2Pixel & 0x00ff0000) >> 16));
                int xGreen = Math.abs(((x1Pixel & 0x0000ff00) >> 8) - ((x2Pixel & 0x0000ff00) >> 8));
                int xBlue = Math.abs((x1Pixel & 0x000000ff) - (x2Pixel & 0x000000ff));

                int yRed = Math.abs(((y1Pixel & 0x00ff0000) >> 16) - ((y2Pixel & 0x00ff0000) >> 16));
                int yGreen = Math.abs(((y1Pixel & 0x0000ff00) >> 8) - ((y2Pixel & 0x0000ff00) >> 8));
                int yBlue = Math.abs((y1Pixel & 0x000000ff) - (y2Pixel & 0x000000ff));*/
                int xRed = (int) Math.pow(((x1Pixel & 0x00ff0000) >> 16) - ((x2Pixel & 0x00ff0000) >> 16),2);
                int xGreen = (int) Math.abs(((x1Pixel & 0x0000ff00) >> 8) - ((x2Pixel & 0x0000ff00) >> 8));
                int xBlue = (int) Math.abs((x1Pixel & 0x000000ff) - (x2Pixel & 0x000000ff));

                int yRed = (int) Math.abs(((y1Pixel & 0x00ff0000) >> 16) - ((y2Pixel & 0x00ff0000) >> 16));
                int yGreen = (int) Math.abs(((y1Pixel & 0x0000ff00) >> 8) - ((y2Pixel & 0x0000ff00) >> 8));
                int yBlue = (int) Math.abs((y1Pixel & 0x000000ff) - (y2Pixel & 0x000000ff));
                

                // We add up all the differences and call that our energy.
                double energy = xRed + xGreen + xBlue + yRed + yGreen + yBlue;

                //energyTable[x][y] = Math.log(energy);
                energyTable[x][y] = energy;
            }
        }

        return energyTable;
    }

    private static int[][] findVerticalSeam(double[][] energyTable) {
        int[][] seam;
        int width = energyTable.length;
        int height = energyTable[0].length;
        // seamDynamic is the table we will use for dynamic programming.
        double[][] seamDynamic = new double[width][height];
        int[][] backtracker = new int[width][height];
        double minimum;
        // vertical seam.
        seam = new int[energyTable[0].length][2];

        // Loops over the energy table and finds the lowest energy path.
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (y == 0) {
                    seamDynamic[x][y] = energyTable[x][y];
                    backtracker[x][y] = -1;
                } else {
                    // every other row.
                    // need to special case the sides.
                    if (x == 0) {
                        minimum = Math.min(seamDynamic[x][y - 1], seamDynamic[x + 1][y - 1]);
                        if (minimum == seamDynamic[x][y - 1]) {
                            // add backtracker.
                            backtracker[x][y] = 1;
                        } else { // x + 1
                            // add backtracker.
                            backtracker[x][y] = 2;
                        }
                    } else if (x == (width - 1)) {
                        minimum = Math.min(seamDynamic[x][y - 1], seamDynamic[x - 1][y - 1]);
                        if (minimum == seamDynamic[x][y - 1]) {
                            // add backtracker.
                            backtracker[x][y] = 1;
                        } else { // x - 1
                            // add backtracker.
                            backtracker[x][y] = 0;
                        }
                    } else {
                        minimum = Math.min(seamDynamic[x - 1][y - 1], Math.min(seamDynamic[x][y - 1], seamDynamic[x + 1][y - 1]));
                        if (minimum == seamDynamic[x - 1][y - 1]) {
                            // add backtracker.
                            backtracker[x][y] = 0;
                        } else if (minimum == seamDynamic[x][y - 1]) {
                            // add backtracker.
                            backtracker[x][y] = 1;
                        } else { // x + 1
                            // add backtracker.
                            backtracker[x][y] = 2;
                        }
                    }
                    seamDynamic[x][y] = energyTable[x][y] + minimum;
                }
            }
        }

        // now that we have computed the paths, we need to backtrace the minimum one.
        // 0 --> x - 1.
        // 1 --> x.
        // 2 --> x + 1.
        // first we need to find the min at the end.
        double min_num = seamDynamic[0][height - 1];
        int min_index = 0;
        for (int x = 0; x < width; x++) {
            if (min_num > seamDynamic[x][height - 1]) {
                min_index = x;
                min_num = seamDynamic[x][height - 1];
            }
        }

        // now that we have the min we need to backtrace it.
        int y_index = height - 1;
        int x_index = min_index;
        seam[y_index][0] = x_index;
        seam[y_index][1] = y_index;
        int backtrack;
        while (y_index > 0) {
            backtrack = backtracker[x_index][y_index];
            if (backtrack != -1) {
                if (backtrack == 0) {
                    x_index = x_index - 1;
                } else if (backtrack == 1) {
                    x_index = x_index;
                } else { // = 2
                    x_index = x_index + 1;
                }
            } else {
                x_index = x_index;
            }
            y_index = y_index - 1;

            seam[y_index][0] = x_index;
            seam[y_index][1] = y_index;
        }
        return seam;
    }

    private static int[][] findHorizontalSeam(double[][] energyTable) {

        int width = energyTable.length;
        int height = energyTable[0].length;
        // seamDynamic is the table we will use for dynamic programming.
        double[][] seamDynamic = new double[width][height];
        int[][] backtracker = new int[width][height];
        double minimum;
        // horizontal seam.
        int[][] seam = new int[energyTable.length][2];

        // Loops over the energy table and finds the lowest energy path.
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (x == 0) {
                    seamDynamic[x][y] = energyTable[x][y];
                    backtracker[x][y] = -1;
                } else {
                    // every other column.
                    // need to special case the top/bottom.
                    if (y == 0) {
                        minimum = Math.min(seamDynamic[x - 1][y], seamDynamic[x - 1][y + 1]);
                        if (minimum == seamDynamic[x - 1][y]) {
                            // add backtracker.
                            backtracker[x][y] = 1;
                        } else { // y + 1
                            // add backtracker.
                            backtracker[x][y] = 2;
                        }
                    } else if (y == (height - 1)) {
                        minimum = Math.min(seamDynamic[x - 1][y], seamDynamic[x - 1][y - 1]);
                        if (minimum == seamDynamic[x - 1][y]) {
                            // add backtracker.
                            backtracker[x][y] = 1;
                        } else { // y - 1
                            // add backtracker.
                            backtracker[x][y] = 0;
                        }
                    } else {
                        minimum = Math.min(seamDynamic[x - 1][y - 1], Math.min(seamDynamic[x - 1][y], seamDynamic[x - 1][y + 1]));
                        if (minimum == seamDynamic[x - 1][y - 1]) {
                            // add backtracker.
                            backtracker[x][y] = 0;
                        } else if (minimum == seamDynamic[x - 1][y]) {
                            // add backtracker.
                            backtracker[x][y] = 1;
                        } else { // y + 1
                            // add backtracker.
                            backtracker[x][y] = 2;
                        }
                    }
                    seamDynamic[x][y] = energyTable[x][y] + minimum;
                }
            }
        }

        // now that we have computed the paths, we need to backtrace the minimum one.
        // 0 --> y - 1.
        // 1 --> y.
        // 2 --> y + 1.
        // first we need to find the min at the end.
        double min_num = seamDynamic[width - 1][0];
        int min_index = 0;
        for (int y = 0; y < height; y++) {
            if (min_num > seamDynamic[width - 1][y]) {
                min_index = y;
                min_num = seamDynamic[width - 1][y];
            }
        }

        // now that we have the min we need to backtrace it.
        int y_index = min_index;
        int x_index = width - 1;
        seam[x_index][0] = x_index;
        seam[x_index][1] = y_index;
        int backtrack;
        while (x_index > 0) {
            backtrack = backtracker[x_index][y_index];
            if (backtrack != -1) {
                if (backtrack == 0) {
                    y_index = y_index - 1;
                } else if (backtrack == 1) {
                    y_index = y_index;
                } else { // = 2
                    y_index = y_index + 1;
                }
            } else {
                y_index = y_index;
            }
            x_index = x_index - 1;

            seam[x_index][0] = x_index;
            seam[x_index][1] = y_index;
        }
        return seam;
    }

    /**
     * findSeam() finds a seam given an energy table and a direction. The seam is
     * the path from bottom to top or left to right with minimum total energy.
     *
     * @param energy table (double[][]) and direction (vertical / horizontal).
     * @return seam (int[x or y][x, y]).
     */
    private static int[][] findSeam(double[][] energyTable, String direction) {
        int[][] seam=new int[1][2];
        if (direction.equals("v")) {
            seam = findVerticalSeam(energyTable);

        } else if (direction.equals("h")) {
            seam = findHorizontalSeam(energyTable);

        } else {

            System.out.println("Invalid direction.");
            System.exit(1);
        }

        return seam;
    }

    /**
     * removeSeam() removes a given seam from an image.
     *
     * @param image, seam[][] and direction (vertical / horizontal).
     * @return carved image.
     */
    private static BufferedImage removeSeam(BufferedImage image, int[][] seam, String direction) {
        BufferedImage newImage;
        int width = image.getWidth();
        int height = image.getHeight();
        if (direction.equals("v")) {
            // vertical seam.
            newImage = new BufferedImage(width - 1, height, BufferedImage.TYPE_INT_ARGB);
        } else {
            // horizontal seam.
            newImage = new BufferedImage(width, height - 1, BufferedImage.TYPE_INT_ARGB);
        }

        // Loops over ever pixel in the original image and copies them over.
        // Do not copy over the pixels in the seam.
        if (direction.equals("v")) {
            // vertical seam.
            for (int y = 0; y < height; y++) {
                boolean shift = false;
                for (int x = 0; x < width; x++) {
                    // Simple loop to check if the pixel is part of the seam or not.
                    boolean inSeam = false;
                    if ((seam[y][0] == x) && (seam[y][1] == y)) {
                        inSeam = true;
                        shift = true;
                    }

                    if (!inSeam) {
                        // pixel not part of the seam, so we add it.
                        int color = image.getRGB(x, y);
                        if (shift) {
                            newImage.setRGB(x - 1, y, color);
                        } else {
                            newImage.setRGB(x, y, color);
                        }
                    }
                }
            }
        } else {
            // horizontal seam.
            for (int x = 0; x < width; x++) {
                boolean shift = false;
                for (int y = 0; y < height; y++) {
                    // Simple loop to check if the pixel is part of the seam or not.
                    boolean inSeam = false;
                    if ((seam[x][0] == x) && (seam[x][1] == y)) {
                        inSeam = true;
                        shift = true;
                    }

                    // this does not work, as we might need to put it at either x-1 or y-1.
                    if (!inSeam) {
                        // pixel not part of the seam, so we add it.
                        if (shift) {
                            newImage.setRGB(x, y - 1, image.getRGB(x, y));
                        } else {
                            newImage.setRGB(x, y, image.getRGB(x, y));
                        }
                    }
                }
            }
        }

        return newImage;
    }

    /**
     * showImage() displays the given image.
     *
     * @param image.
     * @return n/a.
     */
    private static void showImage(BufferedImage image) {
        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        frame.getContentPane().add(new JLabel(new ImageIcon(image)));
        frame.pack();
        frame.setVisible(true);
    }


}