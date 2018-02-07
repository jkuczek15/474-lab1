import java.io.File;

public class NaiveBayes {

	public static final int FILES = 6;
	public static final String USAGE = "Usage: java NaiveBayes <vocabulary> <map> <training_label> <training_data> <testing_label> <testing_data>";
	private static File[] input = new File[FILES];
	
	public static void main(String[] args) {
		// read the input files from command line arguments
		readInput(args);
		
		System.out.println("Successfully read input");
		
	}// end main method
	
	/*
	 * Input order:
	 * 0 - vocabulary.txt
	 * 1 - map.csv
	 * 2 - train_label.csv
	 * 3 - train_data.csv
	 * 4 - test_label.csv
	 * 5 - test_data.csv
	 */
	private static void readInput(String[] args) {
		if(args.length != FILES) {
			System.out.println(USAGE);
			System.exit(0);
		}// end if user didn't pass 6 arguments
		
		for (int i = 0; i < FILES; i++) {
            input[i] = new File(args[i]);
            if(!input[i].isFile()) {
            	System.out.println(args[i] + " could not be opened. Exiting.");
            	System.exit(0);
            }// end if input is not a file
        }// end for loop over each argument
	}// end function readInput

}// end class NaiveBayes
