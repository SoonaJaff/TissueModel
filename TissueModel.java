import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



public class TissueModel {

    //Images are of format given by allowedExtension -
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final long seed = 123;
    private static final int nEpochs = 20 ;
    private static final int batchSize=10;
    private static final int possibleLabel=3;
    
    private static final Random randNumGen = new Random(seed);
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
   

    public static String dataLocalPath="C:\\User\\Sona Jaff\\Desktop\\Report\\Model\\";
    public static void sorttissue() throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String SAMPLE_CSV_FILE_PATH = "C:/Users/Sona Jaff/Desktop/Report/Data/miasData.csv";

	        try (
	            Reader reader = Files.newBufferedReader(Paths.get(SAMPLE_CSV_FILE_PATH));
	            CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT);
	        ) {
	        	String [] filename= new String[322];
	        	String [] Abnormality=new String[322];
	        	int i=0;
	            for (CSVRecord csvRecord : csvParser) {
	                // Accessing Values by Column Index
	                filename [i]= csvRecord.get(0);
	                String pathName="C:/Users/Sona Jaff/Desktop/Report/original/enhance/"+filename[i]+".jpg";
	                File dir1=new File("C:/Users/Sona Jaff/Desktop/Report/original/newdataset/f");
	                if (! dir1.exists()) dir1.mkdir();
	                File dir2=new File("C:/Users/Sona Jaff/Desktop/Report/original/newdataset/g");
	                if (! dir2.exists()) dir2.mkdir();
	                File dir3=new File("C:/Users/Sona Jaff/Desktop/Report/original/newdataset/d");
	                if (! dir3.exists()) dir3.mkdir();
	                
	                Abnormality[i] = csvRecord.get(1);
	                if (Abnormality[i].equals("F"))
	                {
	                	
	                	Mat file= Imgcodecs.imread(pathName, Imgcodecs.IMREAD_COLOR);
	                	pathName="C:/Users/Sona Jaff/Desktop/Report/original/newdataset/f/"+filename[i]+".jpg";
	                	Imgcodecs.imwrite(pathName,file);
	                }
	                else if(Abnormality[i].equals("G"))
	                {
	                	Mat file= Imgcodecs.imread(pathName, Imgcodecs.IMREAD_COLOR);
	                	pathName="C:/Users/Sona Jaff/Desktop/Report/original/newdataset/g/"+filename[i]+".jpg";
	                	Imgcodecs.imwrite(pathName,file);
	                }
	                else
	                {
	                	Mat file= Imgcodecs.imread(pathName, Imgcodecs.IMREAD_COLOR);
	                	pathName="C:/Users/Sona Jaff/Desktop/Report/original/newdataset/d/"+filename[i]+".jpg";
	                	Imgcodecs.imwrite(pathName,file);
	                }
	               
	                if (i<=320)
	                	i++;
	                else
	                	break;
	              
	            }
	        }
	}
    
    public static DataSetIterator loadData(InputSplit data,ParentPathLabelGenerator labelMaker) throws IOException {
    	ImageRecordReader dataReader = new ImageRecordReader(height,width,channels,labelMaker);
    	DataNormalization norm= new NormalizerStandardize();
        dataReader.initialize(data); 
        DataSetIterator dataIter = new RecordReaderDataSetIterator(dataReader,batchSize,1,possibleLabel);
        norm.fit(dataIter);
    	return dataIter;
    }
    @SuppressWarnings("rawtypes")
	public static void train(Logger logger,DataSetIterator trainSet) throws IOException {
    	 ZooModel m=VGG16.builder().build();
         logger.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
         
         ComputationGraph preTrainedNet = (ComputationGraph) m.initPretrained();
         logger.info(preTrainedNet.summary());
         
         FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                 .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                 .updater(new Nesterovs(0.01))
                 .seed(seed) 
                 .build();

         ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
                 .fineTuneConfiguration(fineTuneConf)
                 .setFeatureExtractor("fc2")
                 .removeVertexKeepConnections("predictions")
                 .addLayer("predictions",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                 .nIn(4096).nOut(possibleLabel)//2
                                 .weightInit(WeightInit.XAVIER)
                                 .activation(Activation.SOFTMAX).build(), "fc2")
                 .build();
         
         vgg16Transfer.setListeners(new ScoreIterationListener(1));
         logger.info(vgg16Transfer.summary());
         File locationToSave=new File("TissueModel.zip");
         while (trainSet.hasNext()) {
             DataSet trained = trainSet.next();
             vgg16Transfer.fit(trained);
             
         }
         ModelSerializer.writeModel(vgg16Transfer, locationToSave, true);
         logger.info("Save Trained Model");
    }
    
	
    public static void passes(Logger logger,DataSetIterator trainSet) throws IOException {
    	File locationSaveModel=new File("TissueModel.zip");
    	ComputationGraph model=ModelSerializer.restoreComputationGraph(locationSaveModel);
    		while(trainSet.hasNext()) {
    			DataSet train=trainSet.next();
    			model.fit(train);
    		}
    		ModelSerializer.writeModel(model, locationSaveModel, true);
    		trainSet.reset();
    		logger.info("Model trained.......");
    }
    
    @SuppressWarnings("deprecation")
	public static void evaluate(Logger logger,DataSetIterator testSet) throws IOException {
        File locationToSave=new File("TissueModel.zip");
        ComputationGraph model=ModelSerializer.restoreComputationGraph(locationToSave);
        Evaluation eval = null;
   
       logger.info("Trained Model Loaded");
       while(testSet.hasNext()){
           eval = model.evaluate(testSet);
           logger.info("Evaluation Results.....");
           logger.info(eval.stats()); 
       } 
       logger.info("Model build evaluated");
	}
	
	public static void main (String[] args) throws  IOException  {
    //	BasicConfigurator.configure(); 
		
	//	sorttissue();
		Logger logger = LoggerFactory.getLogger(TissueModel.class);
        File parentDir=new File("C:/Users/Sona Jaff/Desktop/Report/original/newdataset/");
        //Files in directories under the parent dir that have "allowed extensions" split needs a random number generator for reproducibility when splitting the files into train and test
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        //parse the parent dir and use the name of the subdirectories as label/class names
        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadoc for details
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        
        //Split the image files into train and test. Specify the train test split as 80%,20%
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter,0.75, 0.25);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1]; 
        DataSetIterator trainSet= loadData(trainData,labelMaker);
        System.gc();
        DataSetIterator testSet= loadData(testData,labelMaker);
        System.gc();
        train(logger,trainSet);
        System.gc();
        int iter=0;
        System.gc();
        while(iter<nEpochs) {
        	logger.info("Epoch number ("+iter+") started");
        	 passes(logger,trainSet);
        	 iter++;
             System.gc();
        }
        evaluate(logger,testSet);
        
    }
}