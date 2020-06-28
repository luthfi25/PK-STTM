package models;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

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

    public String folderPath;

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

    public List<List<Double>> theta;
    public List<List<Double>> phi;

    public List<List<GT_POINTS>> gtPoints; //ground truth points of knowledge source

    public double left; //Left bound for sigma
    public double right; //right bound for sigma

    public int totalTopics; //T
    public int B = 0; //background knowledge topics

    public List<List<Integer>> corpus_t; //taken from initRandom
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

    public PKLFLDA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
                   double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
                   int inNumIterations, int inTopWords, String inExpName, double inMu, double inSigma, int inApprox,
                   String pathToKS, String pathToGT, double inLeft, double inRight) {
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
        folderPath = "results/";

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
        loadDeltas(pathToKS, pathToGT);
        totalTopics = B;

        initRandom();
        updateN();

        hidden = new boolean[totalTopics];
        for (int i = 0; i < totalTopics; i++) {
            hidden[i] = false;
            visibleTopics.add(i);
        }

        topicProbs = new double[totalTopics];
    }

    public void updateN() {
        n_w = new int[totalTopics][vocabularySize];
        n_w_dot = new int[totalTopics];
        n_d_dot = new int[totalTopics];
        n_d = new int[totalTopics][numDocuments];

        for (int i = 0; i < totalTopics; i++) {
            n_w_dot[i] = 0;
            n_w[i] = new int[vocabularySize];
            n_d_dot[i] = 0;

            for (int j = 0; j < vocabularySize; j++) {
                n_w[i][j] = 0;
            }
            n_d[i] = new int[numDocuments];

            for (int j = 0; j < numDocuments; j++) {
                n_d[i][j] = 0;
            }
        }

        for (int doc = 0; doc < numDocuments; doc++) {
            HashSet<Integer> topics_in_doc = new HashSet<>();

            for (int token = 0; token < corpus.get(doc).size(); token++) {
                int t = corpus_t.get(doc).get(token);
                int w = corpus.get(doc).get(token);
                topics_in_doc.add(t);
                n_w[t][w]++;
                n_d[t][doc]++;
                n_w_dot[t]++;
            }
            for (Integer itr : topics_in_doc) {
                n_d_dot[itr]++;
            }
        }
    }

    public void initRandom() {
        random = new Random();

        corpus_t = new ArrayList<>();
        for (int doc = 0; doc < numDocuments; doc++) {
            List<Integer> topics = new ArrayList<>();
            for (int token = 0; token < corpus.get(doc).size(); token++) {
                int t = random.nextInt(totalTopics);
                topics.add(t);
            }
            corpus_t.set(doc, topics);
        }
    }

    public void loadDeltas(String pathToKnowledgeSource, String pathToGtpoints) {
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
                    srcWords.add(Math.ulp(i * 1.0));
                }

                for (int i = 1; i + 1 < words.length; i++) {
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
        double right_bound = Math.max(mu + 2 * sigma, right_min);

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
    }

    public void loadGTPoints(String pathToGtpoints) {
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

                for (int i = 0; i + 1 < points.length; i++) {
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

    public void inference() {
        System.out.println("Running Gibbs sampling inference: ");

//        int burn = 100;
//        int start_y = totalTopics;

//        int n_prune = (int) Math.ceil(((double)start_y) / ((double)(numIterations-burn)));
//        stats.tot_iteration_time = 0;
        for (int iter = 0; iter <= numIterations; iter++) {
//            stats.assign_correct = 0;
//            stats.assign_total = 0;
//            auto start = high_resolution_clock::now();
//            stats.tot_reg = 0;
//            stats.cnt_reg = 0;
//            stats.tot_model = 0;
//            stats.cnt_model = 0;
//            stats.iteration_time = 0;
//            cout << currentDateTime() << "...SrcLDA.gibbs - begin iter " << iter << "...topics " << visible_topics.size() << endl;

            for (int doc = 0; doc < numDocuments; doc++) {
                for (int token = 0; token < corpus.get(doc).size(); token++) {
                    int newTopic = Sample(doc, token);
                    List<Integer> newDoc = corpus_t.get(doc);
                    newDoc.set(token, newTopic);
                    corpus_t.set(doc, newDoc);
                }
            }

//            stats.iteration_time = duration_cast<milliseconds>(high_resolution_clock::now()-start).count();
//            stats.tot_iteration_time += stats.iteration_time;
//            Display_stats(iter);
        }

        System.out.println("Gibbs sampling done!");
        write();
        return;
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

//        if (n_w_dot[topic] == 0 && visible_topics.size() > K && !hidden[topic] && options.model != bijective) {
//            cout << currentDateTime() << "...removing topic " << topic << endl;
//            Hide_topic(topic);
//        }
//        alpha = options.use_alpha ? options.alpha : ((double)50) / ((double)visible_topics.size());

        topic = pop_sample(w, doc);
        topic = visibleTopics.get(topic);
    //    stats.assign_total++;
    //    int offset = options.model == bijective ? 0 : K;
    //    if (options.use_key && ground_truth[doc][token] + offset == topic) {
    //        stats.assign_correct++;
    //    }
        if (n_d[topic][doc] == 0) {
            n_d_dot[topic]++;
        }
        n_d[topic][doc]++;
        n_w[topic][w]++;
        n_w_dot[topic]++;
        return topic;
    }

    public int pop_sample(int word, int doc){
        for (int i=0; i<visibleTopics.size(); i++) {
            int t = visibleTopics.get(i);
            populate_prob(i, t, word, doc, 0);
        }
        double scale = topicProbs[visibleTopics.size()-1] * random.nextFloat();
        int topic = 0;

        if (topicProbs[0] <= scale) {
            int low = 0;
            int high = visibleTopics.size()-1;
            while (low <= high) {
                if (low == high - 1) { topic = high; break; }
                int mid = (low + high) / 2;
                if (topicProbs[mid] > scale) {
                    high = mid;
                } else{
                    low = mid;
                }
            }
        }

        return topic;
    }

    public void populate_prob(int i, int t, int word, int doc, int start){
//        if (t < K) {
//            pr[i] = (((double) n_w[t][word] + beta) / (((double) n_w_dot[t]) + ((double) V) * beta)) *
//                    (((double) n_d[t][doc] + alpha) / (((double) (corpus[doc].size() - 1)) + ((double) visible_topics.size()) * alpha));
//
//            stats.cnt_reg++;
//            stats.tot_reg += pr[i];
//        }
//        else {
        int b = t;
        double sum = 0.0;
        for (int a=0; a<approx; a++) {
            double delta_i_j = delta_pows[b][word][a];
            double delta_a_j_sum = deltaPowSums[b][a];
            sum += ((((double) n_w[t][word] + delta_i_j) / (((double) n_w_dot[t]) + delta_a_j_sum)) *
                    (((double) n_d[t][doc] + alpha) / (((double) (corpus.get(doc).size() - 1)) + ((double) visibleTopics.size()) * alpha)) *
                    norm[a]);
        }
//
        topicProbs[i] = sum;
//        stats.cnt_model++;
//        stats.tot_model += pr[i];
//        }
        if (i > start) {
            topicProbs[i] += topicProbs[i-1];
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
//        for (int i=0; i<theta.size(); i++) {
//            theta[i].clear();
//        }
//        theta.clear();
        theta = new ArrayList<>();
        int topic_count = visibleTopics.size();
        for (int doc=0; doc<numDocuments; doc++) {
            ArrayList<Double> theta_d = new ArrayList<>();
            for (int t=0; t<totalTopics; t++) {
                double t_theta_d = (((double)n_d[t][doc]) + alpha) / (((double)corpus.get(doc).size()) + (((double)topic_count)*alpha));
                theta_d.add(t_theta_d);
            }
            theta.add(theta_d);
        }
    }

    public void calculate_phi(){
//        for (int i=0; i<phi.size(); i++) {
//            phi[i].clear();
//        }
//        phi.clear();

        phi = new ArrayList<>();
        for (int t=0; t<totalTopics; t++) {
            ArrayList<Double> phi_t = new ArrayList<>();
//            vector<double> phi_t(V);
//            if (t < K) {
//                for (int w=0; w<V; w++) {
//                    phi_t[w] = (((double)n_w[t][w]) + beta)/(((double)n_w_dot[t]) + (((double)V)*beta));
//                }
//            }
//            else {
                int b = totalTopics;
                for (int w=0; w<vocabularySize; w++) {
                    double sum = 0.0;
                    for (int a=0; a<approx; a++) {
                        double delta_i_j = delta_pows[b][w][a];
                        double delta_a_j_sum = deltaPowSums[b][a];
                        sum += (((double) n_w[t][w] + delta_i_j) / (((double) n_w_dot[t]) + delta_a_j_sum))*norm[a];
                    }
                    phi_t.add(sum);
                }
//            }
            phi.add(phi_t);
        }
    }

    public void write_distributions(){
        int top = totalTopics;
        int[] idx = new int[totalTopics];

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
                    + ".theta"));

            for (int doc=0; doc<numDocuments; doc++) {
                for (int t=0; t<totalTopics; t++) {
                    writer.write(theta.get(doc).get(t) + " ");
//                    theta_out << " " << Clean(theta[doc][t]);
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

            for (int t=0; t<totalTopics; t++) {
                if (hidden[t]) { continue; }
                String topic = topicLabels.get(t);
                writer.write(topic + " ");

//            if (options.display.top) {
//                vector<double> phi_t(phi[t]);
//                Sort(phi_t, idx);
//                for (int w=0; w<Min(options.display.n, V); w++) {
//                    string word = options.display.labels ? id_word[idx[w]] : to_string(idx[w]);
//                    phi_out << " " << word << ":" << Clean(phi_t[w]);
//                }
//            }
//            else {
                for (int w=0; w<vocabularySize; w++) {
                    writer.write(phi.get(t).get(w) + " ");
//                    phi_out << " " << Clean(phi[t][w]);
                }

                writer.write("\n");
//            }
//            phi_out << endl;
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}