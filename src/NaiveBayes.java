import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class NaiveBayes {

	// global static variables
	public static final int FILES = 6;
	public static final String USAGE = "Usage: java NaiveBayes <vocabulary> <map> <training_label> <training_data> <testing_label> <testing_data>";
	
	// dynamic variables
	private static File[] input = new File[FILES];
	private static int totalDocuments;
	private static int newsgroups;
	private static int words;
	private static int[] newsgroupDocuments;
	private static int[] newsgroupWords;
	private static int[][] wordCountMatrix;
	private static double[] priors;
	private static double[][] mle;
	private static double[][] laplace;
	private static ArrayList<String> labels;
	
	public static void main(String[] args) throws IOException {
		// read the input files from command line arguments
		readInput(args);
		
		// count the total number of words in all documents
		words = countLines(args[0]);
		
		// count total number of news groups
		newsgroups = countLines(args[1]);
		
		// read all the newsgroups that the documents belong too
		labels = readLabels(input[2]);
		
		// total documents is equal to the number of labels
		totalDocuments = labels.size();
		
		// store total number of documents for each news group (for computing priors)
		newsgroupDocuments = new int[newsgroups];
		
		// create matrix for storing word counts for each newsgroup
		wordCountMatrix = new int[newsgroups][words];
		
		// store total words for each news group
		newsgroupWords = new int[newsgroups];
		
	   /* 
		* Train the model, compute priors and estimators
		*/
		train();
	    		
	   /* 
		* Perform classification tests and report results
		*/
	  	classify(input[3], "training", laplace, "BE");
	  	classify(input[5], "testing", mle, "MLE");
	  	classify(input[5], "testing", laplace, "BE");
	}// end main method
	
	public static void train() throws FileNotFoundException {
	   /* 
		* Loop through training data collecting information
		*/
		HashMap<Integer, Integer> docTracker = new HashMap<Integer, Integer>();
		Scanner sc = new Scanner(input[3]);
		while (sc.hasNextLine()) {
			// loop over all training data
			String[] data = sc.nextLine().split(",");
			
			// grab relevant document information including the 
			// docId, wordId and the wordCount
			int docId = Integer.parseInt(data[0]);
			int wordId = Integer.parseInt(data[1]);
			int wordCount = Integer.parseInt(data[2]);
			
			// grab the newsgroup based on the labels we read earlier
			int newsgroup = Integer.parseInt(labels.get(docId-1));
			
			// update the word matrix to use the new word count
			wordCountMatrix[newsgroup-1][wordId-1] += wordCount;
			
			// count the total number of words for each news group
			newsgroupWords[newsgroup-1] += wordCount;
			
			if(docTracker.get(docId) == null) {
				// count total number of documents for each news group
				docTracker.put(docId, 1);
				newsgroupDocuments[newsgroup-1]++;
			}// end if we haven't counted this document yet
			
		}// end while loop over each line
		sc.close();
		
	   /* 
		* Compute and print out priors
	    */
		priors = new double[newsgroups];
		System.out.println("-------------------------------------------------");
		System.out.println("Class priors:");
		System.out.println("-------------------------------------------------");
		for(int i = 0; i < newsgroups; i++) {
			priors[i] = (double) newsgroupDocuments[i] / totalDocuments;
			System.out.printf("P(Omega = " + (i+1) +") = " + "%.4f\n", priors[i]);
		}// end for loop over newsgroups
		
	   /* 
		* Compute MLE and Bayesian estimators 
		*/
		mle = new double[newsgroups][words];
		laplace = new double[newsgroups][words];
		for(int i = 0; i < newsgroups; i++) {
			for(int j = 0; j < words; j++) {
				mle[i][j] = (double) wordCountMatrix[i][j] / newsgroupWords[i];
				laplace[i][j] = (double) (wordCountMatrix[i][j] + 1) / (newsgroupWords[i] + words);
			}// end for loop computing MLE and Laplace estimations
		}// end for loop over newsgroups
	}// end function train
	
	private static void classify(File file, String type, double[][] estimator, String estimatorType) throws FileNotFoundException {
	   /* 
		* Setup priors for all documents and newsgroups (classification)
		*/
		double[][] estimates = new double[totalDocuments][newsgroups];
		for(int i = 0; i < totalDocuments; i++) {
			
			for(int j = 0; j < newsgroups; j++) {
				estimates[i][j] = Math.log(priors[j]);
			}// end for loop over all documents setting priors
			
		}// end for loop over news groups
		
	   /* 
		* Compute joint logarithm sum using provided estimator
		*/	
		Scanner sc = new Scanner(file);
		while (sc.hasNextLine()) {
			String[] data = sc.nextLine().split(",");
			
			// grab the id of this word in the document
			int docId = Integer.parseInt(data[0]);
			int wordId = Integer.parseInt(data[1]);
		
			for(int j = 0; j < newsgroups; j++) {
				// compute the estimate using joint product
				double estimate = Math.log(estimator[j][wordId-1]);
				
				// sum up all our estimate values
				estimates[docId-1][j] += estimate;
			}// end for loop over all documents setting priors
			
		}// end while loop over each line
		sc.close();
		
	   /* 
		* Compute maximum category estimate for each document
		*/
		int[] classified = new int[totalDocuments];
		int[] classCorrect = new int[newsgroups];
		int[][] confusionMatrix = new int[newsgroups][newsgroups];
		int totalCorrect = 0;
		for(int i = 0; i < totalDocuments; i++) {
			// loop through all documents computing max
			double max = estimates[i][0];
			int maxIndex = 0;
			
			for(int j = 1; j < newsgroups; j++) {
				if(estimates[i][j]  > max) {
					max = estimates[i][j];
					maxIndex = j;
				}// end if new max estimate
			}// end for loop over newsgroups
			
			// place this document into a category
			classified[i] = maxIndex+1;
			
			// determine if we classified the document correctly
			int correctGroup = Integer.parseInt(labels.get(i));
			confusionMatrix[correctGroup-1][classified[i]-1]++;
			if(classified[i] == correctGroup) {
				// we correctly classified this document
				totalCorrect++;
				classCorrect[correctGroup-1]++;
			}// end if we correctly classified this document
			
		}// end for loop over all documents
		
	   reportAccuracy(totalCorrect, classCorrect, confusionMatrix, type, estimatorType);
	}// end function classify
	
	public static void reportAccuracy(int totalCorrect, int[] classCorrect, int[][] confusionMatrix, String type, String estimatorType) {
	   /* 
		* Report accuracy results with data
		*/
		double accuracy = (double) totalCorrect / totalDocuments;
		System.out.println("-------------------------------------------------");
		System.out.println("Performance on "+type+" data with "+estimatorType+":");
		System.out.println("-------------------------------------------------");
		System.out.printf("Overall Accuracy = %.4f\n", accuracy);
		System.out.println("Class Accuracy:");
		for(int i = 0; i < newsgroups; i++) {
			double classAccuracy = (double) classCorrect[i] / newsgroupDocuments[i];
			System.out.printf("Group " + (i+1) + ": %.4f\n", classAccuracy);
		}// end for loop printing out class accuracy
		// print the confusion matrix
		System.out.println("Confusion matrix:");
		final PrettyPrinter printer = new PrettyPrinter(System.out);
		printer.print(confusionMatrix);
	}// end function reportAccuracy
	
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
	
	/*
	 * Count all the lines in a file efficiently.
	 * This function requires that the input file ends with a new line
	 */
	public static int countLines(String filename) throws IOException {
	    InputStream is = new BufferedInputStream(new FileInputStream(filename));
	    try {
	        byte[] c = new byte[1024];
	        int count = 0;
	        int readChars = 0;
	        boolean empty = true;
	        while ((readChars = is.read(c)) != -1) {
	            empty = false;
	            for (int i = 0; i < readChars; ++i) {
	                if (c[i] == '\n') {
	                    ++count;
	                }// end if we've reached the end of the line
	            }// end for loop over characters
	        }// end while loop over lines
	        return (count == 0 && !empty) ? 1 : count;
	    } finally {
	        is.close();
	    }// end try-catch block
	}// end function countLines
	
	private static ArrayList<String> readLabels(File file) throws FileNotFoundException {
		Scanner sc = new Scanner(file);
		ArrayList<String> lines = new ArrayList<String>();
		while (sc.hasNextLine()) {
			lines.add(sc.nextLine());
		}// end while loop over each line
		sc.close();
		return lines;
	}// end function readLabels
	
}// end class NaiveBayes
