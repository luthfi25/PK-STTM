package models;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class PKLFLDA {
    public int numTopics;
    public double alpha;
    public double beta;
    public double lambda;
    public int numInitIterations;
    public int numIterations;
    public int topWords;
    public String expName;
    public double mu;
    public double sigma;
    public int approx;

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID

    public List<List<Integer>> corpus; // Word ID-based corpus
    public List<List<Integer>> topicAssignments; // Topics assignments for words
    // in the corpus
    public int numDocuments; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus

    public int vocabularySize; // The number of word types in the corpus

    public List<String> topicLabels; //labels for topic absorbed from knowledge source
    public List<List<Double>> deltas; //given knowledge-source topic, give the word occurence

    public List<List<GT_POINTS>> gtPoints; //ground truth points of knowledge source

    public double left; //Left bound for sigma
    public double right; //right bound for sigma

    public PKLFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
                   double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
                   int inNumIterations, int inTopWords, String inExpName, double inMu, double inSigma, int inApprox,
                   String pathToKS, String pathToGT, double inLeft, double inRight){
        System.out.println("Corpus: " + pathToCorpus);
        System.out.println("Vector File: " + pathToWordVectorsFile);
        System.out.println("Num. Topics: " + inNumTopics);
        System.out.println("Alpha: " + inAlpha);
        System.out.println("Beta: " + inBeta);
        System.out.println("Lambda: " + inLambda);
        System.out.println("Num. Init. Iterations: " + inNumInitIterations);
        System.out.println("Num. Iterations: " + inNumInitIterations);
        System.out.println("Num. Top Words: " + inTopWords);
        System.out.println("Exp. Name: " + inExpName);
        System.out.println("Mean Source Distribution: " + inMu);
        System.out.println("Stdev Source Distribution: " + inSigma);
        System.out.println("Approximation Step: " + inApprox);
        System.out.println("Knowledge Source path: " + pathToKS);

        // Assigning Variable
        numTopics = inNumTopics;
        alpha = inAlpha;
        beta = inBeta;
        lambda = inLambda;
        numInitIterations = inNumInitIterations;
        numIterations = inNumIterations;
        topWords = inTopWords;
        expName = inExpName;
        mu = inMu;
        sigma = inSigma;
        approx = inApprox;
        left = inLeft;
        right = inRight;

        //Assign word2Id and id2word
        BufferedReader br = null;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathToCorpus));
            for (String doc; (doc = br.readLine()) != null;) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split("\\s+");
                List<Integer> document = new ArrayList<Integer>();

                for (String word : words) {
                    if (word2IdVocabulary.containsKey(word)) {
                        document.add(word2IdVocabulary.get(word));
                    }
                    else {
                        indexWord += 1;
                        word2IdVocabulary.put(word, indexWord);
                        id2WordVocabulary.put(indexWord, word);
                        document.add(indexWord);
                    }
                }

                numDocuments++;
                numWordsInCorpus += document.size();
                corpus.add(document);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size();
        loadDeltas(pathToKS, pathToGT);
    }

    public void loadDeltas(String pathToKnowledgeSource, String pathToGtpoints){
        for (int i=0; i < numTopics; i++){
            topicLabels.add(String.valueOf(i));
        }

        //Assign topic labels and deltas from knowledge source
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToKnowledgeSource));
            for (String topic; (topic = br.readLine()) != null;) {

                if (topic.trim().length() == 0)
                    continue;

                String[] words = topic.trim().split("\\s+");
                topicLabels.add(words[0]);

                List<Double> srcWords = new ArrayList<Double>();
                for (int i = 0; i < vocabularySize; i++){
                    srcWords.add(Math.ulp(i * 1.0));
                }

                for(int i = 1; i+1 < words.length; i++){
                    if (word2IdVocabulary.containsKey(words[i])) {
                        int wordID = word2IdVocabulary.get(words[i]);
                        double newCount = srcWords.get(wordID);
                        newCount += Integer.parseInt(words[i+1]);
                        srcWords.set(wordID, newCount);
                    }
                }

                deltas.add(srcWords);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        int B = deltas.size();
        double[][] deltaPowSums = new double[B][];

        for (int i=0; i < B; i++){
            deltaPowSums[i] = new double[approx];
        }

        double[] lambdas;
        if (sigma == 0.0){
            lambdas = new double[1];
        } else {
            lambdas = new double[approx];
        }

        double left_max = left;
        double right_min = right;

        if(pathToGtpoints != ""){
            loadGTPoints(pathToGtpoints);
            left_max = 0.0;
            right_min = 1.0;
        }

        double left_bound = Math.max(mu - 2*sigma, left_max);
        double right_bound = Math.max(mu + 2*sigma, right_min);

        double interval = (right_bound - left_bound) / (Double.valueOf(approx-1));
        if (approx < 2) {
            interval = 0;
        }

        for (int a = 0; a < approx; a++){
            lambdas[a] = left_bound + a * interval;
        }

        double[] norm = new double[approx];
        double sum = 0.0;
        for (int a = 0; a < approx; a++){
            if(sigma == 0.0){
                norm[a] = 1.0;
                sum += 1.0;
                break;
            }

            double x = lambdas[a];
            double density = normal(x, mu, sigma);
            norm[a] = density;
            sum += density;
        }

//        for (int a=0; a<A; a++) {
//            norm[a] = norm[a]/sum;
//        }
//
//        delta_pows = new double**[B];
//
//        for (int i=0; i<B; i++) {
//            delta_pows[i] = new double*[V];
//            double sum[A];
//            for (int a = 0; a < A; a++) {
//                sum[a] = 0.0;
//            }
//            for (int j=0; j<V; j++) {
//                delta_pows[i][j] = new double[A];
//                for (int a = 0; a < A; a++) {
//                    double mapped = options.use_gtpoints ? gt_points.map(i, lambdas[a]) : lambdas[a];
//                    double val = pow(deltas[i][j], mapped);
//                    delta_pows[i][j][a] = val;
//                    sum[a] += val;
//                }
//            }
//            for (int a=0; a<A; a++) {
//                delta_pow_sums[i][a] = sum[a];
//            }
//        }
    }

    public void loadGTPoints(String pathToGtpoints){
        gtPoints.clear();

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToGtpoints));
            for (String line; (line = br.readLine()) != null;) {
                if(line.trim().length() == 0){
                    continue;
                }

                String[] points = line.trim().split("\\s+");
                List<GT_POINTS> gtPointsLine = new ArrayList<GT_POINTS>();

                for(int i = 0; i+1 < points.length; i++){
                    double x = Double.parseDouble(points[i]);
                    double y = Double.parseDouble(points[i+1]);
                    GT_POINTS gtPoint = new GT_POINTS(x, y);
                    gtPointsLine.add(gtPoint);
                }

                gtPoints.add(gtPointsLine);
            }
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    public double normal(double x, double mu, double sigma){
        double xMinusMuSquared = (x - mu) * (x - mu);
        double twoSigmaSquared = (sigma * sigma) * 2.0;
        double exponent = -1.0 * xMinusMuSquared / twoSigmaSquared;
        double eExponent = Math.exp(exponent);
        double sqrtTwoSigmaSquaredPi = Math.sqrt(twoSigmaSquared * Math.PI);
        double result = (1.0 / sqrtTwoSigmaSquaredPi) * eExponent;
        return result;
    }
}
