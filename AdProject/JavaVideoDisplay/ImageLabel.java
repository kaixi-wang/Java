import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.util.ArrayList;

import javax.swing.*;

public class ImageLabel extends JLabel {
    private static final long serialVersionUID = 1L;
    public ImageIcon mImage;
    public ImageLabel(ImageIcon image) {
        super(image);
        this.mImage = image;
    }

    @Override
	public  void paintComponent(Graphics g) {
        super.paintComponent(g);
		Image image = mImage.getImage();
		g.drawImage(image,0,0,this);
	}

	public void SetImage(ImageIcon image) {
        mImage = image;
        this.repaint();
	}
}