// package org.wikijava.sound.playWave;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.Clip;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.UnsupportedAudioFileException;
import javax.sound.sampled.DataLine.Info;

/**
 * 
 * <Replace this with a short description of the class.>
 * 
 * @author Giulio
 */
public class PlaySound {

    private InputStream waveStream;

	private final int EXTERNAL_BUFFER_SIZE = 524288; // 128Kb
    // private final int EXTERNAL_BUFFER_SIZE = 16000; // 1 frame
	AudioInputStream audioInputStream = null;
	public Clip dataLine = null;

    /**
     * CONSTRUCTOR
     */
    public PlaySound(InputStream waveStream) {
		this.waveStream = waveStream;
    }

    public void play() throws PlayWaveException {
		try {
			//audioInputStream = AudioSystem.getAudioInputStream(this.waveStream);
			
			//add buffer for mark/reset support, modified by Jian
			InputStream bufferedIn = new BufferedInputStream(this.waveStream);
			audioInputStream = AudioSystem.getAudioInputStream(bufferedIn);
		} catch (UnsupportedAudioFileException e1) {
			throw new PlayWaveException(e1);
		} catch (IOException e1) {
			throw new PlayWaveException(e1);
		}

		// Obtain the information about the AudioInputStream
		AudioFormat audioFormat = audioInputStream.getFormat();
		Info info = new Info(Clip.class, audioFormat);

		// opens the audio channel
		try {
			dataLine = (Clip) AudioSystem.getLine(info);
			dataLine.open(audioInputStream);
		} catch (Exception e1) {
			throw new PlayWaveException(e1);
		}

		// Starts the music :P
		dataLine.loop(1);

		int readBytes = 0;
		byte[] audioBuffer = new byte[this.EXTERNAL_BUFFER_SIZE];
	}
	
}
