
import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.image.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;

import javax.swing.*;

public class ImageDisplay {
	JFrame frame;
	ImageLabel lbIm1;
	ArrayList<BufferedImage> images = new ArrayList<BufferedImage>();
	// BufferedImage[] images;
	int width = 480;
	int height = 270;

	// used to calculate frame rate etc.
	long start_time = 0;
	long last_time = 0;
	long elpase_time = 0;

	// used for file IO etc
	long current_frame;
	String video_path;
	String audio_path;
	long video_length = 0;

	boolean isPlaying = true;
	PlaySound playSound;

	/**
	 * Read Image RGB Reads the image of given width and height at the given imgPath
	 * into the provided BufferedImage.
	 */
	private void readImageRGB(int width, int height, String imgPath, BufferedImage img) {
		try {
			int frameLength = width * height * 3;
			File file = new File(imgPath);
			this.video_length = file.length() / frameLength;
			RandomAccessFile raf = new RandomAccessFile(file, "r");
			raf.seek(current_frame * width * height * 3);

			long len = frameLength;
			byte[] bytes = new byte[(int) len];
			raf.read(bytes);
			int ind = 0;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					byte r = bytes[ind];
					byte g = bytes[ind + height * width];
					byte b = bytes[ind + height * width * 2];

					int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
					// int pix = ((a << 24) + (r << 16) + (g << 8) + b);
					img.setRGB(x, y, pix);
					ind++;
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void Initialize(String[] args) {
		this.current_frame = 0;
		// Read a parameter from command line

		this.video_path = args[0];
		this.audio_path = args[1];

		// Read in the specified image
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		readImageRGB(width, height, args[0], img);

		FileInputStream inputStream;
		try {
			inputStream = new FileInputStream(audio_path);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}
		// initializes the playSound Object
		
		
		// try {
		// playSound.PlayFrame();
		// } catch (PlayWaveException e) {
		// 	e.printStackTrace();
		// 	return;
		// }

		// long last_time = System.nanoTime();
		// Use label to display the image
		frame = new JFrame();
		GridBagLayout gLayout = new GridBagLayout();
		frame.getContentPane().setLayout(gLayout);

		lbIm1 = new ImageLabel(new ImageIcon(img));

		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.5;
		c.gridx = 0;
		c.gridy = 0;

		frame.getContentPane().add(lbIm1, c);
		
		JButton pauseButton = new JButton("Pause");
		pauseButton.addActionListener((ActionListener) new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				isPlaying = false;
				// System.out.println(playSound.dataLine.getMicrosecondPosition());
				playSound.dataLine.stop();
			}
		});
		frame.getContentPane().add(pauseButton);

		JButton startButton = new JButton("Play");
		startButton.addActionListener((ActionListener) new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				isPlaying = true;
				// System.out.println(playSound.dataLine.getMicrosecondPosition());
				playSound.dataLine.start();
			}
		});
		frame.getContentPane().add(startButton);

		frame.pack();
		
		// Initial Tick Time
		start_time = System.nanoTime();
		last_time = System.nanoTime();

		playSound = new PlaySound(inputStream);
		Runnable myRunnable =
			new Runnable(){
				public void run(){
					try {
						playSound.play();
					} catch (PlayWaveException e) {
						e.printStackTrace();
						return;
					}
				}
			};
		Thread thread = new Thread(myRunnable);
		thread.start();
		frame.setVisible(true);
	}

	public void update() {
		if (current_frame < video_length - 1) {
			long time = System.nanoTime();
			if (isPlaying) {
				long delta_time = time - last_time;
				elpase_time += delta_time;
				long old_frame = this.current_frame;
				this.current_frame = (elpase_time / (1000000000 / 30));
				
				BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
				readImageRGB(width, height, video_path, img);
				
				lbIm1.SetImage(new ImageIcon(img));
				frame.repaint();
			}
			last_time = time;
			// System.out.println(delta_time);
		} else {
			playSound.dataLine.stop();
		}
	}
}