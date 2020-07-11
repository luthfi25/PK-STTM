package models;

import utility.FuncUtils;
import utility.LBFGS;
import utility.Parallel;
import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.types.MatrixOps;
import utility.TopicVectorOptimizer;

import java.io.*;
import java.util.*;

public class PKLFLDA {
    public int numTopics;
    public double alpha;
    public double beta;
    public double alphaSum; // alpha * numTopics
    public double betaSum; // beta * vocabularySize
    public double lambda;
    public int numInitIterations;
    public int numIterations;
    public int topWords;
    public String expName;
    public double mu;
    public double sigma;
    public int approx;

    public String folderPath;
    public String vectorFilePath;

    public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID
    // given a word
    public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word
    // given an ID

    public List<List<Integer>> corpus; // Word ID-based corpus
    // in the corpus
    public int numDocuments; // Number of documents in the corpus
    public int numWordsInCorpus; // Number of words in the corpus

    public int vocabularySize; // The number of word types in the corpus

    public List<String> topicLabels; //labels for topic absorbed from knowledge source
    public List<List<Double>> deltas; //given knowledge-source topic, give the word occurence

    public List<List<Double>> theta;
    public List<List<Double>> phi;

    public List<List<GT_POINTS>> gtPoints; //ground truth points of knowledge source

    public double left; //Left bound for sigma
    public double right; //right bound for sigma

    public int totalTopics; //T
    public int B = 0; //background knowledge topics

    public List<List<Integer>> corpus_t; //taken from initRandom, topicAssignment in LFLDA
    public Random random; //taken from initRandom
    public List<Integer> visibleTopics; //from load

    public double[] topicProbs; //pr

    //from updateN
    public int[][] n_w;
    public int[] n_w_dot;
    public int[] n_d_dot;
    public int[][] n_d;

    //from load_delta
    double[][][] delta_pows;
    double[][] deltaPowSums;
    double[] norm;
    boolean[] hidden;

    // numDocuments * numTopics matrix
    // Given a document: number of its words assigned to each topic
    public int[][] docTopicCount;
    // Number of words in every document
    public int[] sumDocTopicCount;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type generated from the topic by
    // the Dirichlet multinomial component
    public int[][] topicWordCountLDA;
    // Total number of words generated from each topic by the Dirichlet
    // multinomial component
    public int[] sumTopicWordCountLDA;
    // numTopics * vocabularySize matrix
    // Given a topic: number of times a word type generated from the topic by
    // the latent feature component
    public int[][] topicWordCountLF;
    // Total number of words generated from each topic by the latent feature
    // component
    public int[] sumTopicWordCountLF;

    //vectors
    public double[][] wordVectors; // Vector representations for words
    public double[][] topicVectors;// Vector representations for topics
    public int vectorSize; // Number of vector dimensions
    public double[][] dotProductValues;
    public double[][] expDotProductValues;
    public double[] sumExpValues; // Partition function values

    //optimizetopicvectors
    public final double l2Regularizer = 0.01; // L2 regularizer value for learning topic vectors
    public final double tolerance = 0.05; // Tolerance value for LBFGS convergence

    //separate usage topicprobs and multipros
    double[] sampleProbs;

    public PKLFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
                   double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
                   int inNumIterations, int inTopWords, String inExpName, double inMu, double inSigma, int inApprox,
                   String pathToKS, String pathToGT, double inLeft, double inRight) throws Exception {
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
        folderPath = "results/";
        vectorFilePath = pathToWordVectorsFile;

        File dir = new File(folderPath);
        if (!dir.exists())
            dir.mkdir();

        word2IdVocabulary = new HashMap<>();
        id2WordVocabulary = new HashMap<>();
        corpus = new ArrayList<>();
        topicLabels = new ArrayList<>();
        deltas = new ArrayList<>();
        gtPoints = new ArrayList<>();

        //Assign word2Id and id2word
        BufferedReader br = null;
        try {
            int indexWord = -1;
            br = new BufferedReader(new FileReader(pathToCorpus));
            for (String doc; (doc = br.readLine()) != null; ) {

                if (doc.trim().length() == 0)
                    continue;

                String[] words = doc.trim().split("\\s+");
                List<Integer> document = new ArrayList<Integer>();

                for (String word : words) {
                    if (word2IdVocabulary.containsKey(word)) {
                        document.add(word2IdVocabulary.get(word));
                    } else {
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
        } catch (Exception e) {
            e.printStackTrace();
        }

        vocabularySize = word2IdVocabulary.size();


        //begin source-LDA part
        loadDeltas(pathToKS, pathToGT);
        totalTopics = B + numTopics;

        //begin lf lda part
        docTopicCount = new int[numDocuments][numTopics];
        sumDocTopicCount = new int[numDocuments];
        topicWordCountLDA = new int[numTopics][vocabularySize];
        sumTopicWordCountLDA = new int[numTopics];
        topicWordCountLF = new int[numTopics][vocabularySize];
        sumTopicWordCountLF = new int[numTopics];

        topicProbs = new double[totalTopics];
        for (int i = 0; i < totalTopics; i++) {
            topicProbs[i] = 1.0 / numTopics;
        }

        alphaSum = numTopics * alpha;
        betaSum = vocabularySize * beta;

        try{
            readWordVectorsFile(vectorFilePath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        topicVectors = new double[numTopics][vectorSize];
        dotProductValues = new double[numTopics][vocabularySize];
        expDotProductValues = new double[numTopics][vocabularySize];
        sumExpValues = new double[numTopics];

//         initialize();
        initRandom();
        updateN();

        visibleTopics = new ArrayList<>();
        hidden = new boolean[totalTopics];
        for (int i = 0; i < totalTopics; i++) {
            hidden[i] = false;
            visibleTopics.add(i);
        }

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

//        topicProbs = new double[totalTopics];
    }

    public void readWordVectorsFile(String pathToWordVectorsFile)
            throws Exception
    {
        System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile
                + "...");

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToWordVectorsFile));
            String[] elements = br.readLine().trim().split("\\s+");
            vectorSize = elements.length - 1;
            wordVectors = new double[vocabularySize][vectorSize];
            String word = elements[0];
            if (word2IdVocabulary.containsKey(word)) {
                for (int j = 0; j < vectorSize; j++) {
                    wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
                }
            }
            for (String line; (line = br.readLine()) != null;) {
                elements = line.trim().split("\\s+");
                word = elements[0];
                if (word2IdVocabulary.containsKey(word)) {
                    for (int j = 0; j < vectorSize; j++)
                        wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        for (int i = 0; i < vocabularySize; i++) {
            if (MatrixOps.absNorm(wordVectors[i]) == 0.0) {
                System.out.println("The word \"" + id2WordVocabulary.get(i)
                        + "\" doesn't have a corresponding vector!!!");
//                throw new Exception();
            }
        }
    }

    public void updateN() {
        n_w = new int[totalTopics][vocabularySize];
        n_w_dot = new int[totalTopics];
        n_d_dot = new int[numTopics];
        n_d = new int[numTopics][numDocuments];

        for (int i = 0; i < totalTopics; i++) {
            if (i < numTopics){
                // document counts is limited only to numtopics
                n_d_dot[i] = 0;

                n_d[i] = new int[numDocuments];
                for (int j = 0; j < numDocuments; j++) {
                    n_d[i][j] = 0;
                }
            }

            n_w_dot[i] = 0;
            n_w[i] = new int[vocabularySize];

            for (int j = 0; j < vocabularySize; j++) {
                n_w[i][j] = 0;
            }
        }

        for (int doc = 0; doc < numDocuments; doc++) {
            HashSet<Integer> topics_in_doc = new HashSet<>();

            for (int token = 0; token < corpus.get(doc).size(); token++) {
                int subt = corpus_t.get(doc).get(token);
                int t = subt % numTopics;
                int w = corpus.get(doc).get(token);
                topics_in_doc.add(t);
                n_w[subt][w]++;
                n_w_dot[subt]++;
                n_d[t][doc]++;
            }
            for (Integer itr : topics_in_doc) {
                n_d_dot[itr]++;
            }
        }
    }

    public void initRandom() {
        corpus_t = new ArrayList<>();
        for (int doc = 0; doc < numDocuments; doc++) {
            List<Integer> topics = new ArrayList<>();
            for (int token = 0; token < corpus.get(doc).size(); token++) {
                int wordId = corpus.get(doc).get(token);
                int t = FuncUtils.nextDiscrete(topicProbs);
                int subt = t % numTopics;

                if (t == subt) { // Generated from the latent feature component
                    topicWordCountLF[subt][wordId] += 1;
                    sumTopicWordCountLF[subt] += 1;
                }
                else {// Generated from the Dirichlet multinomial component (source LDA)
                    topicWordCountLDA[subt][wordId] += 1;
                    sumTopicWordCountLDA[subt] += 1;
                }

                docTopicCount[doc][subt] += 1;
                sumDocTopicCount[doc] += 1;

                topics.add(t);
            }
            corpus_t.add(topics);
        }
    }

    public void loadDeltas(String pathToKnowledgeSource, String pathToGtpoints) {
        System.out.println("Loading deltas...");
        for (int i = 0; i < numTopics; i++) {
            topicLabels.add(String.valueOf(i));
        }

        //Assign topic labels and deltas from knowledge source
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToKnowledgeSource));
            for (String topic; (topic = br.readLine()) != null; ) {

                if (topic.trim().length() == 0)
                    continue;

                String[] words = topic.trim().split("\\s+");
                topicLabels.add(words[0]);

                List<Double> srcWords = new ArrayList<Double>();
                for (int i = 0; i < vocabularySize; i++) {
                    srcWords.add(Math.ulp(1.0));
                }

                for (int i = 1; i + 1 < words.length; i += 2) {
                    if (word2IdVocabulary.containsKey(words[i])) {
                        int wordID = word2IdVocabulary.get(words[i]);
                        double newCount = srcWords.get(wordID);
                        newCount += Integer.parseInt(words[i + 1]);
                        srcWords.set(wordID, newCount);
                    }
                }

                deltas.add(srcWords);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        B = deltas.size();
        deltaPowSums = new double[B][];

        for (int i = 0; i < B; i++) {
            deltaPowSums[i] = new double[approx];
        }

        double[] lambdas;
        if (sigma == 0.0) {
            lambdas = new double[1];
        } else {
            lambdas = new double[approx];
        }

        double left_max = left;
        double right_min = right;

        if (pathToGtpoints != "") {
            loadGTPoints(pathToGtpoints);
            left_max = 0.0;
            right_min = 1.0;
        }

        double left_bound = Math.max(mu - 2 * sigma, left_max);
        double right_bound = Math.min(mu + 2 * sigma, right_min);

        double interval = (right_bound - left_bound) / (Double.valueOf(approx - 1));
        if (approx < 2) {
            interval = 0;
        }

        for (int a = 0; a < approx; a++) {
            lambdas[a] = left_bound + a * interval;
        }

        norm = new double[approx];
        double sum = 0.0;
        for (int a = 0; a < approx; a++) {
            if (sigma == 0.0) {
                norm[a] = 1.0;
                sum += 1.0;
                break;
            }

            double x = lambdas[a];
            double density = normal(x, mu, sigma);
            norm[a] = density;
            sum += density;
        }

        for (int a = 0; a < approx; a++) {
            norm[a] = norm[a] / sum;
        }

        delta_pows = new double[B][vocabularySize][approx];

        System.out.println("mapping gt points...");
        for (int i = 0; i < B; i++) {
            delta_pows[i] = new double[vocabularySize][approx];
            double[] subSum = new double[approx];
            for (int a = 0; a < approx; a++) {
                subSum[a] = 0.0;
            }
            for (int j = 0; j < vocabularySize; j++) {
                delta_pows[i][j] = new double[approx];
                for (int a = 0; a < approx; a++) {
                    double mapped = mapGTPoints(i, lambdas[a]);
                    double val = Math.pow(deltas.get(i).get(j), mapped);
                    delta_pows[i][j][a] = val;
                    subSum[a] += val;
                }
            }
            for (int a = 0; a < approx; a++) {
                deltaPowSums[i][a] = subSum[a];
            }
        }
        System.out.println("finished mapping gt points!");

        System.out.println("Finished loading deltas!");
    }

    public void loadGTPoints(String pathToGtpoints) {
        System.out.println("Loading gt points...");
        gtPoints.clear();

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(pathToGtpoints));
            for (String line; (line = br.readLine()) != null; ) {
                if (line.trim().length() == 0) {
                    continue;
                }

                String[] points = line.trim().split("\\s+");
                List<GT_POINTS> gtPointsLine = new ArrayList<GT_POINTS>();

                for (int i = 0; i + 1 < points.length; i = i+2) {
                    double x = Double.parseDouble(points[i]);
                    double y = Double.parseDouble(points[i + 1]);
                    GT_POINTS gtPoint = new GT_POINTS(x, y);
                    gtPointsLine.add(gtPoint);
                }

                gtPoints.add(gtPointsLine);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Finished loading gt points!");
    }

    public double mapGTPoints(int id, double x) {
        try {
            int low = gtPoints.get(id).size() - 1;
            int max = gtPoints.get(id).size() - 1;
            int high = 0;

            while (high <= low) {
                int mid = (low + high) / 2;
                if (gtPoints.get(id).get(mid).getX() == x) {
                    return gtPoints.get(id).get(mid).getY();
                }
                if (gtPoints.get(id).get(low).getX() == x) {
                    return gtPoints.get(id).get(low).getY();
                }
                if (gtPoints.get(id).get(high).getX() == x) {
                    return gtPoints.get(id).get(high).getY();
                }
                if (low == high + 1) {
                    mid = high;
                }
                if (gtPoints.get(id).get(mid).getX() > x && x > gtPoints.get(id).get(Math.min(mid + 1, max)).getX()) {
                    double x_s = gtPoints.get(id).get(Math.min(mid + 1, max)).getX();
                    double x_l = gtPoints.get(id).get(mid).getX();
                    double y_s = gtPoints.get(id).get(Math.min(mid + 1, max)).getY();
                    double y_l = gtPoints.get(id).get(mid).getY();
                    double x_gap = x_l - x_s;
                    double y_gap = y_l - y_s;
                    double frac = (x - x_s) / x_gap;
                    return y_s + frac * y_gap;
                } else if (gtPoints.get(id).get(mid).getX() > x) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            throw new Exception("Error load_deltas: high > low");
        } catch (Exception e) {
            e.printStackTrace();
        }
        return 0.0;
    }

    public double normal(double x, double mu, double sigma) {
        double xMinusMuSquared = (x - mu) * (x - mu);
        double twoSigmaSquared = (sigma * sigma) * 2.0;
        double exponent = -1.0 * xMinusMuSquared / twoSigmaSquared;
        double eExponent = Math.exp(exponent);
        double sqrtTwoSigmaSquaredPi = Math.sqrt(twoSigmaSquared * Math.PI);
        double result = (1.0 / sqrtTwoSigmaSquaredPi) * eExponent;
        return result;
    }

    public void inference() throws IOException {
        System.out.println("Running Gibbs sampling inference: ");

        for (int iter = 1; iter <= numInitIterations; iter++) {

            System.out.println("\tInitial sampling iteration: " + (iter));

            sampleSingleInitialIteration();
        }

        for (int iter = 1; iter <= numIterations; iter++) {
            System.out.println("iteration " + iter + "....");
            optimizeTopicVectors();
            for (int doc = 0; doc < numDocuments; doc++) {
                for (int token = 0; token < corpus.get(doc).size(); token++) {
                    sampleSingleIteration(doc, token);
                }
            }
        }

        System.out.println("Gibbs sampling done!");
        write();
        return;
    }

    public void optimizeTopicVectors()
    {
        System.out.println("\t\tEstimating topic vectors ...");
        sumExpValues = new double[numTopics];
        dotProductValues = new double[numTopics][vocabularySize];
        expDotProductValues = new double[numTopics][vocabularySize];

        Parallel.loop(numTopics, new Parallel.LoopInt()
        {
            @Override
            public void compute(int topic)
            {
                int rate = 1;
                boolean check = true;
                while (check) {
                    double l2Value = l2Regularizer * rate;
                    try {
                        TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
                                topicVectors[topic], topicWordCountLF[topic], wordVectors, l2Value);

                        Optimizer gd = new LBFGS(optimizer, tolerance);
                        gd.optimize(600);
                        optimizer.getParameters(topicVectors[topic]);
                        sumExpValues[topic] = optimizer.computePartitionFunction(
                                dotProductValues[topic], expDotProductValues[topic]);
                        check = false;

                        if (sumExpValues[topic] == 0 || Double.isInfinite(sumExpValues[topic])) {
                            double max = -1000000000.0;
                            for (int index = 0; index < vocabularySize; index++) {
                                if (dotProductValues[topic][index] > max)
                                    max = dotProductValues[topic][index];
                            }
                            for (int index = 0; index < vocabularySize; index++) {
                                expDotProductValues[topic][index] = Math
                                        .exp(dotProductValues[topic][index] - max);
                                sumExpValues[topic] += expDotProductValues[topic][index];
                            }
                        }
                    }
                    catch (InvalidOptimizableException e) {
                        e.printStackTrace();
                        check = true;
                    }
                    rate = rate * 10;
                }
            }
        });
    }

    public void sampleSingleInitialIteration()
    {
        for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
            int docSize = corpus.get(dIndex).size();
            for (int wIndex = 0; wIndex < docSize; wIndex++) {

                int word = corpus.get(dIndex).get(wIndex);// wordID
                int subtopic = corpus_t.get(dIndex).get(wIndex);
                int topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] -= 1;
                if (subtopic == topic) { // LF(w|t) + LDA(t|d)
                    if(topicWordCountLF[topic][word] == 0){
                        System.out.println("ZERO! "+topic+ " "+word);
                    }
                    topicWordCountLF[topic][word] -= 1;
                    sumTopicWordCountLF[topic] -= 1;
                }
                else { // LDA(w|t) + LDA(t|d)
                    topicWordCountLDA[topic][word] -= 1;
                    sumTopicWordCountLDA[topic] -= 1;
                }

                //source-lda
                n_w[subtopic][word] = Math.max(n_w[subtopic][word]-1, 0);
                n_w_dot[subtopic] = Math.max(n_w_dot[subtopic]-1, 0);
                n_d[topic][dIndex] = Math.max(n_d[topic][dIndex]-1, 0);
                if (n_d[topic][dIndex] == 0) {
                    n_d_dot[topic]--;
                }

                // Sample a pair of topic z and binary indicator variable s
                for (int tIndex = 0; tIndex < numTopics; tIndex++) {

                    topicProbs[tIndex] = (docTopicCount[dIndex][tIndex] + alpha) * lambda
                            * (topicWordCountLF[tIndex][word] + beta)
                            / (sumTopicWordCountLF[tIndex] + betaSum);

                    //Source-LDA
                    double sum = 0.0;
                    for (int a=0; a<approx; a++) {
                        double delta_i_j = delta_pows[topic][word][a];
                        double delta_a_j_sum = deltaPowSums[topic][a];
                        sum += ((((double) n_w[tIndex + numTopics][word] + delta_i_j) / (((double) n_w_dot[tIndex + numTopics]) + delta_a_j_sum)) *
                                ((double) n_d[tIndex][dIndex] + alpha) * norm[a] * (1-lambda));
                    }
                    topicProbs[tIndex + numTopics] = sum;

                }
                subtopic = FuncUtils.nextDiscrete(topicProbs);
                topic = subtopic % numTopics;

                docTopicCount[dIndex][topic] += 1;
                if (topic == subtopic) {
                    topicWordCountLF[topic][word] += 1;
                    sumTopicWordCountLF[topic] += 1;
                }
                else {
                    topicWordCountLDA[topic][word] += 1;
                    sumTopicWordCountLDA[topic] += 1;
                }

                //source-lda
                if (n_d[topic][dIndex] == 0) {
                    n_d_dot[topic]++;
                }
                n_d[topic][dIndex]++;
                n_w[topic][word]++;
                n_w_dot[topic]++;

                // Update topic assignments
                corpus_t.get(dIndex).set(wIndex, subtopic);
            }

        }
    }

    public void sampleSingleIteration(int doc, int token) {
        // Get current word
        int word = corpus.get(doc).get(token);// wordID
        int subtopic = corpus_t.get(doc).get(token);
        int topic = subtopic % numTopics;

        docTopicCount[doc][topic] -= 1;
        if (subtopic == topic) {
            topicWordCountLF[topic][word] -= 1;
            sumTopicWordCountLF[topic] -= 1;
        } else {
            topicWordCountLDA[topic][word] -= 1;
            sumTopicWordCountLDA[topic] -= 1;
        }

        //source-lda
        n_d[topic][doc] = Math.max(n_d[topic][doc]-1, 0);
        n_w[subtopic][word] = Math.max(n_w[subtopic][word]-1, 0);
        n_w_dot[subtopic] = Math.max(n_w_dot[subtopic]-1, 0);
        if (n_d[topic][doc] == 0) {
            n_d_dot[topic]--;
        }

        // Sample a pair of topic z and binary indicator variable s
        for (int tIndex = 0; tIndex < numTopics; tIndex++) {

            topicProbs[tIndex] = (docTopicCount[doc][tIndex] + alpha) * lambda
                    * expDotProductValues[tIndex][word] / sumExpValues[tIndex];

            //Source-LDA
            double sum = 0.0;
            for (int a=0; a<approx; a++) {
                double delta_i_j = delta_pows[topic][word][a];
                double delta_a_j_sum = deltaPowSums[topic][a];
                sum += ((((double) n_w[tIndex + numTopics][word] + delta_i_j) / (((double) n_w_dot[tIndex + numTopics]) + delta_a_j_sum)) *
                        ((double) n_d[tIndex][doc] + alpha) * norm[a] * (1-lambda));
            }
            topicProbs[tIndex + numTopics] = sum;

        }
        subtopic = FuncUtils.nextDiscrete(topicProbs);
        topic = subtopic % numTopics;

        docTopicCount[doc][topic] += 1;
        if (subtopic == topic) {
            topicWordCountLF[topic][word] += 1;
            sumTopicWordCountLF[topic] += 1;
        } else {
            topicWordCountLDA[topic][word] += 1;
            sumTopicWordCountLDA[topic] += 1;
        }

        //source-lda
        if (n_d[topic][doc] == 0) {
            n_d_dot[topic]++;
        }
        n_d[topic][doc]++;
        n_w[subtopic][word]++;
        n_w_dot[subtopic]++;

        // Update topic assignments
        corpus_t.get(doc).set(token, subtopic);
    }

    public int Sample(int doc, int token) {
        int topic = corpus_t.get(doc).get(token);
        int w = corpus.get(doc).get(token);
        n_d[topic][doc] = Math.max(n_d[topic][doc]-1, 0);
        n_w[topic][w] = Math.max(n_w[topic][w]-1, 0);
        n_w_dot[topic] = Math.max(n_w_dot[topic]-1, 0);
        if (n_d[topic][doc] == 0) {
            n_d_dot[topic]--;
        }

        topic = pop_sample(w, doc);

        if (n_d[topic][doc] == 0) {
            n_d_dot[topic]++;
        }
        n_d[topic][doc]++;
        n_w[topic][w]++;
        n_w_dot[topic]++;
        return topic;
    }

    public int pop_sample(int word, int doc){
        sampleProbs = new double[numTopics];

        for (int i=numTopics; i<totalTopics; i++) {
            populate_prob(i, i, word, doc, 0);
        }
        double scale = sampleProbs[totalTopics-1] * random.nextFloat();
        int topic = 0;

        if (sampleProbs[0] <= scale) {
            int low = 0;
            int high = numTopics-1;
            while (low <= high) {
                if (low == high - 1) { topic = high; break; }
                int mid = (low + high) / 2;
                if (sampleProbs[mid] > scale) {
                    high = mid;
                } else{
                    low = mid;
                }
            }
        }

        return topic+numTopics;
    }

    public void populate_prob(int i, int t, int word, int doc, int start){
        //add filter lflda
        int b = t - numTopics;
        double sum = 0.0;
        for (int a=0; a<approx; a++) {
            double delta_i_j = delta_pows[b][word][a];
            double delta_a_j_sum = deltaPowSums[b][a];
            sum += ((((double) n_w[t][word] + delta_i_j) / (((double) n_w_dot[t]) + delta_a_j_sum)) *
                    (((double) n_d[t][doc] + alpha) / (((double) (corpus.get(doc).size() - 1)) + ((double) totalTopics) * alpha)) *
                    norm[a]);
        }

        topicProbs[i] = sum;
        sampleProbs[b] = sum;

        if (b > start) {
            sampleProbs[b] += sampleProbs[b-1];
        }

        return;
    }

    public void write(){
        calculate_theta();
        calculate_phi();
        write_distributions();
        return;
    }

    public void calculate_theta(){
        theta = new ArrayList<>();
        int topic_count = numTopics;
        for (int doc=0; doc<numDocuments; doc++) {
            ArrayList<Double> theta_d = new ArrayList<>();
            for (int t=0; t<numTopics; t++) {
                double t_theta_d_old = (((double)n_d[t][doc]) + alpha) / (((double)corpus.get(doc).size()) + (((double)topic_count)*alpha));
                double t_theta_d = (double)Math.round(t_theta_d_old * 1000000d) / 1000000d; // rounding float
                theta_d.add(t_theta_d);
            }
            theta.add(theta_d);
        }
    }

    public void calculate_phi(){
        phi = new ArrayList<>();
        for (int t=0; t<numTopics; t++) {
            ArrayList<Double> phi_t = new ArrayList<>();
            for (int w = 0; w < vocabularySize; w++) {
                double pro_lf = lambda * expDotProductValues[t][w] / sumExpValues[t];
                double pro_pk = (1-lambda);

                double sum = 0.0;
                for (int a=0; a<approx; a++) {
                    double delta_i_j = delta_pows[t][w][a];
                    double delta_a_j_sum = deltaPowSums[t][a];
                    sum += (((double) n_w[t][w] + delta_i_j) / (((double) n_w_dot[t]) + delta_a_j_sum))*norm[a];
                }

                pro_pk *= sum;
                phi_t.add(pro_lf + pro_pk);
            }
            phi.add(phi_t);
        }
    }

    public void write_distributions(){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                    + ".theta"));

            for (int doc=0; doc<numDocuments; doc++) {
                for (int t=0; t<totalTopics; t++) {
                    writer.write(theta.get(doc).get(t) + " ");
                }
                writer.write("\n");
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                    + ".phi"));

            for (int t=0; t<numTopics; t++) {
                if (hidden[t]) { continue; }
                String topic = topicLabels.get(t + numTopics);
                writer.write(topic + " ");

                for (int w=0; w<vocabularySize; w++) {
                    writer.write(phi.get(t).get(w) + " ");
                }

                writer.write("\n");
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                    + ".topWords"));

            for (int tIndex = 0; tIndex < numTopics; tIndex++) {
                writer.write("Topic" + topicLabels.get(tIndex + numTopics) + ":");

                Map<Integer, Double> topicWordProbs = new TreeMap<Integer, Double>();
                for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
                    double pro_lf = lambda * expDotProductValues[tIndex][wIndex] / sumExpValues[tIndex];
                    double pro_pk = (1-lambda);

                    double sum = 0.0;
                    for (int a=0; a<approx; a++) {
                        double delta_i_j = delta_pows[tIndex][wIndex][a];
                        double delta_a_j_sum = deltaPowSums[tIndex][a];
                        sum += (((double) n_w[tIndex][wIndex] + delta_i_j) / (((double) n_w_dot[tIndex]) + delta_a_j_sum))*norm[a];
                    }

                    pro_pk *= sum;
                    topicWordProbs.put(wIndex, pro_lf + pro_pk);
                }
                topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs);

                Set<Integer> mostLikelyWords = topicWordProbs.keySet();
                int count = 0;
                for (Integer index : mostLikelyWords) {
                    if (count < topWords) {
                        writer.write( id2WordVocabulary.get(index) + " ");
                        count += 1;
                    }
                    else {
                        writer.write("\n");
                        break;
                    }
                }
            }

            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String args[])
            throws Exception
    {
        return;
    }
}