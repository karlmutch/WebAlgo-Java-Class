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

public class CopticTrainer {

    public static void main(String[] args) throws Exception {
	
		File inputDirectory = new File(args[0]);

        String corpusName = args[1];
        int maxNGram = Integer.valueOf(args[2]);
        double lambdaFactor = Double.valueOf(args[3]);
        int numChars = Integer.valueOf(args[4]);
        File modelDir = new File(args[5]);

        NGramProcessLM lm
            = new NGramProcessLM(maxNGram, numChars, lambdaFactor);
        WeightedEditDistance distance
            = CompiledSpellChecker.TOKENIZING;
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
            String refLine;
            while ((refLine = bufReader.readLine()) != null) {
                String trimmedLine = refLine.trim() + " ";
                String normalizedLine = trimmedLine.replaceAll("\\s+"," ");
                trainer.handle(normalizedLine);
            }
			Streams.closeInputStream(fileIn);
        }

        File modelFile
            = new File(modelDir, "words-coptic.CompiledSpellChecker");
        System.out.println("Saving Spell Checker to " + modelFile);

		FileOutputStream fileOut = null;
        BufferedOutputStream bufOut = null;
        ObjectOutputStream objOut = null;
		
        try {
            fileOut = new FileOutputStream(modelFile);
            bufOut = new BufferedOutputStream(fileOut);
            objOut = new ObjectOutputStream(bufOut);
            trainer.compileTo(objOut);
        } finally {
            Streams.closeOutputStream(objOut);
            Streams.closeOutputStream(bufOut);
            Streams.closeOutputStream(fileOut);
        }

    }
}
