import com.aliasi.classify.PrecisionRecallEvaluation;

import com.aliasi.lm.NGramProcessLM;

import com.aliasi.spell.CompiledSpellChecker;
import com.aliasi.spell.FixedWeightEditDistance;
import com.aliasi.spell.TrainSpellChecker;
import com.aliasi.spell.WeightedEditDistance;

import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Compilable;
import com.aliasi.util.Files;
import com.aliasi.util.ObjectToCounterMap;
import com.aliasi.util.Streams;
import com.aliasi.util.Strings;
import com.aliasi.util.Tuple;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;


public class CopticTokens {

    CompiledSpellChecker mSpellChecker;

    PrecisionRecallEvaluation mBreakEval = new PrecisionRecallEvaluation();
    PrecisionRecallEvaluation mChunkEval = new PrecisionRecallEvaluation();

    ObjectToCounterMap<Integer> mReferenceLengthHistogram = new ObjectToCounterMap<Integer>();
    ObjectToCounterMap<Integer> mResponseLengthHistogram = new ObjectToCounterMap<Integer>();

    Set<Character> mTrainingCharSet = new HashSet<Character>();
    Set<Character> mTestCharSet = new HashSet<Character>();
    Set<String> mTrainingTokenSet = new HashSet<String>();
    Set<String> mTestTokenSet = new HashSet<String>();

    // parameter values
	String mInputDirectory;
    String mCorpusName;
    File mOutputFile;

    File mKnownToksFile;
    Writer mOutputWriter;
    int mMaxNGram;
    double mLambdaFactor;
    int mNumChars;
    int mMaxNBest;

    String mCharEncoding = "UTF-8";

    public String mModelFileName = null;
	
    static void addTokChars(Set<Character> charSet,
                            Set<String> tokSet,
                            String line) {
        if (line.indexOf("  ") >= 0) {
                String msg = "Illegal double space.\n"
                    + "    line=/" + line + "/";
                throw new RuntimeException(msg);
        }
        String[] toks = line.split("\\s+");
        for (int i = 0; i < toks.length; ++i) {
            String tok = toks[i];
            if (tok.length() == 0) {
                String msg = "Illegal token length= 0\n"
                    + "    line=/" + line + "/";
                throw new RuntimeException(msg);
            }
            tokSet.add(tok);
            for (int j = 0; j < tok.length(); ++j) {
                charSet.add(Character.valueOf(tok.charAt(j)));
            }
        }
    }

    static <E> void prEval(String evalName,
                           Set<E> refSet,
                           Set<E> responseSet,
                           PrecisionRecallEvaluation eval) {
        for (E e : refSet)
            eval.addCase(true,responseSet.contains(e));

        for (E e : responseSet)
            if (!refSet.contains(e))
                eval.addCase(false,true);
    }

    public static void main(String[] args) {
        try {
            new ChineseTokens(args).run();
        } catch (Throwable t) {
            System.out.println("EXCEPTION IN RUN:");
            t.printStackTrace(System.out);
        }
    }

    // size of (set1 - set2)
    static <E> int sizeOfDiff(Set<E> set1, Set<E> set2) {
        HashSet<E> diff = new HashSet<E>(set1);
        diff.removeAll(set2);
        return diff.size();
    }

    static String[] extractLines(InputStream in, Set<Character> charSet, Set<String> tokenSet,
                                 String encoding)
        throws IOException {

        ArrayList<String> lineList = new ArrayList<String>();
        InputStreamReader reader = new InputStreamReader(in);
        BufferedReader bufReader = new BufferedReader(reader);
        String refLine;
        while ((refLine = bufReader.readLine()) != null) {
            String trimmedLine = refLine.trim() + " ";
            String normalizedLine = trimmedLine.replaceAll("\\s+"," ");
            lineList.add(normalizedLine);
            addTokChars(charSet, tokenSet, normalizedLine);
        }
        return lineList.toArray(new String[0]);
    }

    static Set<Integer> getSpaces(String xs) {
        Set<Integer> breakSet = new HashSet<Integer>();
        int index = 0;
        for (int i = 0; i < xs.length(); ++i)
            if (xs.charAt(i) == ' ')
                breakSet.add(Integer.valueOf(index));
            else
                ++index;
        return breakSet;
    }

    static Set<Tuple<Integer>>
        getChunks(String xs,
                  ObjectToCounterMap<Integer> lengthCounter) {
        Set<Tuple<Integer>> chunkSet = new HashSet<Tuple<Integer>>();
        String[] chunks = xs.split(" ");
        int index = 0;
        for (int i = 0; i < chunks.length; ++i) {
            int len = chunks[i].length();
            Tuple<Integer> chunk
                = Tuple.create(Integer.valueOf(index),
                               Integer.valueOf(index+len));
            chunkSet.add(chunk);
            index += len;
            lengthCounter.increment(Integer.valueOf(len));
        }
        return chunkSet;
    }

    public CopticTokens(String[] args) 
	{
		mInputDirectory = args[0];
        mCorpusName = args[1];
        mOutputFile = new File(mCorpusName + ".segments");
        mKnownToksFile = new File(mCorpusName + ".knownWords");
        mMaxNGram = Integer.valueOf(args[2]);
        mLambdaFactor = Double.valueOf(args[3]);
        mNumChars = Integer.valueOf(args[4]);
        mMaxNBest = Integer.valueOf(args[5]);

        System.out.println("    Data Input Directory=" + mInputTrainerDirectory);
        System.out.println("    Corpus Name=" + mCorpusName);
        System.out.println("    Output File Name=" + mOutputFile);
        System.out.println("    Known Tokens File Name=" + mKnownToksFile);
        System.out.println("    Max N-gram=" + mMaxNGram);
        System.out.println("    Lambda factor=" + mLambdaFactor);
        System.out.println("    Num chars=" + mNumChars);
        System.out.println("    Max n-best=" + mMaxNBest);
    }

    void run() throws ClassNotFoundException, IOException {
        compileSpellChecker();
        testSpellChecker();
        printResults();
    }

    void compileSpellChecker() throws IOException, ClassNotFoundException 
	{
		File inputDirectory = new File(mInputDirectory + "/Training");
	
		NGramProcessLM lm
            = new NGramProcessLM(mMaxNGram, mNumChars, mLambdaFactor);
        WeightedEditDistance distance
            = new CompiledSpellChecker.TOKENIZING;
        TrainSpellChecker trainer
            = new TrainSpellChecker(lm, distance, null);

		for (String aFile : inputDirectory.list()) {

			if (new File(aFile).isDirectory()) {
				continue;
			}

			FileInputStream fileIn = new FileInputStream(aFile);

			System.out.println("Reading Training Data from " + aFile);

            InputStreamReader reader
                = new InputStreamReader(fileIn, Strings.UTF8);

            BufferedReader bufReader = new BufferedReader(reader);
			String [] lines = extractLines(bufReader, mTrainingCharSet,
                                           mTrainingTokenSet, mCharEncoding);
			for ( String aLine : lines) {
				trainer.handle(aLine);
			}
			Streams.closeInputStream(fileIn);
        }

        mSpellChecker = (CompiledSpellChecker) AbstractExternalizable.compile(trainer);

        mSpellChecker.setAllowInsert(true);
        mSpellChecker.setAllowMatch(true);
        mSpellChecker.setAllowDelete(false);
        mSpellChecker.setAllowSubstitute(false);
        mSpellChecker.setAllowTranspose(false);
        mSpellChecker.setNumConsecutiveInsertionsAllowed(1);
        mSpellChecker.setNBest(mMaxNBest);
    }

    void testSpellChecker() throws IOException 
	{
		File inputDirectory = new File(mInputDirectory + "/Testing");
	
        OutputStream out = new FileOutputStream(mOutputFile);
        mOutputWriter = new OutputStreamWriter(out, Strings.UTF8);

		for (String aFile : inputDirectory.list()) {

			if (new File(aFile).isDirectory()) {
				continue;
			}


			FileInputStream fileIn = new FileInputStream(aFile);

			System.out.println("Reading Testing Data from " + aFile);
			
            InputStreamReader reader
                = new InputStreamReader(fileIn, Strings.UTF8);

            BufferedReader bufReader = new BufferedReader(reader);
			String [] lines = extractLines(bufReader, mTestCharSet,
                                           mTestTokenSet, mCharEncoding);
   			for ( String aLine : lines) {

				String response = mSpellChecker.didYouMean(aLine.replaceAll(" ","")) + ' ';

				mOutputWriter.write(response);
				mOutputWriter.write("\n");

				Set<Integer> refSpaces = getSpaces(reference);
				Set<Integer> responseSpaces = getSpaces(response);
				prEval("Break Points", refSpaces, responseSpaces, mBreakEval);

				Set<Tuple<Integer>> refChunks
					= getChunks(reference, mReferenceLengthHistogram);
				Set<Tuple<Integer>> responseChunks
					= getChunks(response, mResponseLengthHistogram);
				prEval("Chunks", refChunks, responseChunks, mChunkEval);
			}
			FileInputStream.close(fileIn);
		}
    }

    void printResults() throws IOException {
        StringBuilder sb = new StringBuilder();
        Iterator<String> it = mTrainingTokenSet.iterator();
        while (it.hasNext()) {
            sb.append(it.next());
            sb.append('\n');
        }
        Files.writeStringToFile(sb.toString(),mKnownToksFile,
                                Strings.UTF8);

        System.out.println("  Found " + mTestTokenSet.size() + " test tokens.");
        System.out.println("  Found "
                           + sizeOfDiff(mTestTokenSet, mTrainingTokenSet)
                           + " unknown test tokens.");
        System.out.print("    Found " + mTestCharSet.size() + " test characters.");
        System.out.println("  Found "
                           + sizeOfDiff(mTestCharSet, mTrainingCharSet)
                           + " unknown test characters.");

        System.out.println("\nReference/Response Token Length Histogram");
        System.out.println("Length, #REF, #RESP, Diff");
        for (int i = 1; i < 10; ++i) {
            Integer iObj = Integer.valueOf(i);
            int refCount = mReferenceLengthHistogram.getCount(iObj);
            int respCount = mResponseLengthHistogram.getCount(iObj);
            int diff = respCount-refCount;
            System.out.println("    " + i
                               + ", " + refCount
                               + ", " + respCount
                               + ", " + diff);
        }

        System.out.println("Scores");
        System.out.println("  EndPoint:"
                           + " P=" + mBreakEval.precision()
                           + " R=" + mBreakEval.recall()
                           + " F=" + mBreakEval.fMeasure());
        System.out.println("     Chunk:"
                           + " P=" + mChunkEval.precision()
                           + " R=" + mChunkEval.recall()
                           + " F=" + mChunkEval.fMeasure());
    }

    public static void main(String[] args) {
        try {
            new CopticTokens(args).run();
        } catch (Throwable thrown) {
            thrown.printStackTrace(System.err);
        }
    }

}
