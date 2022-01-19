import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class ReadMNIST {

	public static void main(String[] args) {
		
		try {
			
			if(args.length != 2) {
				throw new Exception("Invalid arguments.");
			}
			
			File mnist = new File(args[0]);
			BufferedInputStream mbis = new BufferedInputStream(new FileInputStream(mnist));
			
			byte[] magic = new byte[Integer.BYTES];
			mbis.read(magic);
			byte[] imgCountBytes = new byte[Integer.BYTES];
			mbis.read(imgCountBytes);
			byte[] rowCountBytes = new byte[Integer.BYTES];
			mbis.read(rowCountBytes);
			byte[] colCountBytes = new byte[Integer.BYTES];
			mbis.read(colCountBytes);
			
			File labels = new File(args[1]);
			BufferedInputStream lbis = new BufferedInputStream(new FileInputStream(labels));
			
			magic = new byte[Integer.BYTES];
			lbis.read(magic);
			imgCountBytes = new byte[Integer.BYTES];
			lbis.read(imgCountBytes);
			
			int imgCount = ByteBuffer.wrap(imgCountBytes).getInt();
			int rowCount = ByteBuffer.wrap(rowCountBytes).getInt();
			int colCount = ByteBuffer.wrap(colCountBytes).getInt();
			List<double[]> images = new ArrayList<>();
			for(int i = 0; i < imgCount; i++) {
				double[] image = new double[(rowCount * colCount) + 1];
				image[0] = lbis.read();
				for(int j = 1; j <= rowCount * colCount; j++) {
					image[j] = mbis.read();
				}
				images.add(image);
			}
			
			mbis.close();
			lbis.close();
			
		} catch(IOException e) {
			e.printStackTrace();
		} catch(Exception e) {
			e.printStackTrace();
		}		
	}

}
