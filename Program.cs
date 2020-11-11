using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using Spire.Doc;
using Spire.Doc.Documents;
using System.Text;
using Spire.Pdf;
using F23.StringSimilarity;
using System.Collections.Immutable;

namespace PDFTextExtract
{
    class Program
    {
        static void Main(string[] args)
        {
            //data holder
            var pdf_text = new object();

            MLContext mlContext = new MLContext();
            var emptylist = new List<Input>();
            //convert list into dataview
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptylist);

            //create the transformation pipeline
            var textPipeline = mlContext.Transforms.Text.NormalizeText("Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens",
                    Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.English))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens",
                WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding));
            //fit estimator to data
            var textTransformer = textPipeline.Fit(emptyDataView);
            //prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<Input, TransformedText>(textTransformer);


            //using word document
            //Load Document
            
            Document document1 = new Document();
            document1.LoadFromFile(@"C:/Users/ben/Documents/attachments/CV1.docx");

            //using pdf document
            //Load Document
            Spire.Pdf.PdfDocument pdoc = new Spire.Pdf.PdfDocument();
            pdoc.LoadFromFile(@"C:/Users/ben/Documents/Volvo_Resume.pdf");
            
            //Initialzie StringBuilder Instance for pdf document
            StringBuilder sbpdf = new StringBuilder();
            //Extract text from all pages
            foreach(PdfPageBase page in pdoc.Pages)
            {
                sbpdf.Append(page.ExtractText());

            }
            

            //Initialzie StringBuilder Instance for word document
            StringBuilder sb = new StringBuilder();
            
            //Extract Text from Word and Save to StringBuilder Instance
            foreach (Section section in document1.Sections)  
            { 
                foreach (Paragraph paragraph in section.Paragraphs)
                    
                {  
                    sb.AppendLine(paragraph.Text); 
                }
                
            }


            //get the input text from pdf file
            var data = new Input() { Text = pdf_text.ToString() };

            //text from the doc file
            var data1 = new Input() { Text = sb.ToString()};

            //text from the doc file
            //var data2 = new Input() { Text = sbpdf.ToString() };

            //Test data with job specification
            var test_data = new Input() { Text = "Sofware Developer" };

            //call the prediction API
            var prediction = predictionEngine.Predict(data);


            //predict test data
            var test_pred = predictionEngine.Predict(test_data);


            //Print the length off embedding vector for PDF File
            Console.WriteLine($"Number of Features: {prediction.Features.Length}");
            
            //Print embedding vector
            Console.Write("Features: ");
            foreach (var f in prediction.Features)
            {
                Console.Write($"{f:F4} ");
                
            }
            

            //Randomly Retreive some of the feature gotten from the word embeddings vector
            float word = (float)prediction.Features.GetValue(40);
            float word2 = (float)prediction.Features.GetValue(89);
            float word3 = (float)prediction.Features.GetValue(100);
            float word4 = (float)prediction.Features.GetValue(3);
            float word5 = (float)prediction.Features.GetValue(9);
            float word6 = (float)prediction.Features.GetValue(0);
            float word7 = (float)prediction.Features.GetValue(78);
            float word8 = (float)prediction.Features.GetValue(20);

            //Randomly Retrieve features from the test data
            float test_word = (float)test_pred.Features.GetValue(0);
            float test_word2 = (float)test_pred.Features.GetValue(39);
            float test_word3 = (float)test_pred.Features.GetValue(45);
            float test_word4 = (float)test_pred.Features.GetValue(99);
            float test_word5 = (float)test_pred.Features.GetValue(30);
            float test_word6 = (float)test_pred.Features.GetValue(24);
            float test_word7 = (float)test_pred.Features.GetValue(26);
            

            //calculating mikownski distance
            //create a dictionary to add the word embeddings vector
            Dictionary<string, List<double>> map = new Dictionary<string, List<double>>();
            List<double> listA = new List<double>(new double[] { 
                (double)word,(double)word2,(double)word3,(double)word4,(double)word5,(double)word6, (double)word7 });
            map.Add("Feature1", listA);

            //a testlist to hold the job description word embedding vector
            List<double> testList = new List<double>(new double[] { 
                (double)test_word, (double)test_word2, (double)test_word3, (double)test_word4, (double)test_word5, (double)test_word6, (double)test_word7 });

            string similar_word = mostSimilarKey(testList, map);
            Console.WriteLine("Most similar key is: " + similar_word);

            


        }

        public static string mostSimilarKey(List<double> testList, Dictionary<string, List<double>> map)
        {
            double minDifference = Double.MaxValue;
            string ans = "";
            foreach (var pair in map)
            {
                double absoluteDifference = getAbsoluteDifference(testList, pair.Value);
                if (absoluteDifference < minDifference)
                {
                    minDifference = absoluteDifference;
                    ans = pair.Key;
                }
            }
            return ans;
        }

        public static double getAbsoluteDifference(List<double> testList, List<double> list)
        {
            double absoluteDifference = 0.0;
            for (int i = 0; i < testList.Count; ++i)
            {
                absoluteDifference += Math.Abs(testList[i] - list[i]);
            }
            return absoluteDifference;
        }
    }
}
