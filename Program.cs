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
using Accord.Math;
using System.IO;
using Microsoft.ML.Data;

namespace PDFTextExtract
{
    class Program
    {
        private static readonly Cosine cacheCosine = new Cosine();
        private static MLContext mlContext;
        private static string dir_path = "C:/Users/ben/Documents/attachments/CV1.docx";
        private static string desc_path = "C:/Users/ben/Documents/attachments/" +
                "job descriptions/Developer.txt";
        private static string doc_text;
        private static string pdf_text;
        static void Main(string[] args)
        {
            //Get the file extension to determine if it is a doc file or pdf file
            FileInfo file_type = new FileInfo(dir_path);
            if (file_type.Extension == ".docx")
            {
                Console.WriteLine("Adding: {0}...", Path.GetFileName(dir_path));
                doc_text = GetTextDoc(dir_path);

            }
            else if (file_type.Extension == ".pdf")
            {
                Console.WriteLine("Adding: {0}...", Path.GetFileName(dir_path));
                
            }
            else
            {
                Console.WriteLine("{0} is not a valid file or directory.", dir_path);
            }

            //Initialize an ML context
            mlContext = new MLContext();

            //Load the extracted text from doc file to Input class
            var doc_data = new Input() { Text = doc_text };
            

            //Create an empty list class to hold all the inputs from the file
            var emptylist = new List<Input>();
            //convert list into dataview
            var emptyDataView = mlContext.Data.LoadFromEnumerable(emptylist);

            //Initialize the word embeddings pipeline 
            /**Word embeddings model uses Glove which is a pretrained model with 50Dimensions**/
            var wordemb_pipeline = TextPipeline();

            //Fit the pipeline with the empty data view 
            var textTransformer = wordemb_pipeline.Fit(emptyDataView);


            //prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<Input, TransformedText>(textTransformer);


            //Job description data to be used for cosine similarity analysis
            string job_text = GetTxtText(desc_path);

            /**Take two documents and compare the similarity between them using cosine similarity
             * F23.StringSimilarity libary
             * It takes two strings and calculates the similarity between the two strings
             * For this comparison we will use one CV and compare it with a job descrption for a developer job**/
            var threshold = cacheCosine.Similarity(job_text, doc_text);
            Console.WriteLine();
            Console.WriteLine("Similarity score");
            Console.WriteLine(threshold);

            /**Similarity score is measured between 0 and 1 
             * when the score is closer to one the similarity between the words is high
             * when the score is closer to zero the similarity between the words is low**/
            if(threshold < 0.5)
            {
                Console.WriteLine("CV doesn't match requirements expected");
            }
            else
            {
                Console.WriteLine("CV match requirements expected");
            }


            //Job description data
            var job_data = new Input() { Text = job_text.ToString() };

            //call the prediction API
            var prediction = predictionEngine.Predict(doc_data);

            //predict job description data
            var job_pred = predictionEngine.Predict(job_data);

            //Print the length off embedding vector
            Console.WriteLine($"Number of Features(CV): {prediction.Features.Length}");
            Console.WriteLine($"Number of Features(job description): {job_pred.Features.Length}");

            //Print embedding vector
            Console.Write("Features: ");
            foreach (var f in prediction.Features)
            {
                //Console.Write($"{f:F4} ");

            }

            /**Calculate the cosine similarity using the features from Glove word embeddings
             * vector1 has features from the CV text document
             * vector2 has features from the job description document**/

            double[] vector1 = new double[(prediction.Features.Length)];
            Array.Copy(prediction.Features, vector1, prediction.Features.Length);
            Console.WriteLine();
            Console.WriteLine(vector1[7]);

            double[] vector2 = new double[(job_pred.Features.Length)];
            Array.Copy(job_pred.Features, vector2, job_pred.Features.Length);
            Console.WriteLine();
            Console.WriteLine(vector2[9]);

            //Calculate the similarity using using the Similarity function
            var similarity_vec = Similarity(vector1, vector2);
            Console.WriteLine();
            Console.WriteLine("Cosine similarity score with two vectors");
            Console.WriteLine(similarity_vec);

 
        }


        //Cosine similarity function
        public static double Similarity(double[] x, double[] y)
        {
            double sum = 0;
            double p = 0;
            double q = 0;

            for (int i = 0; i < x.Length; i++)
            {
                sum += x[i] * y[i];
                p += x[i] * x[i];
                q += y[i] * y[i];
            }

            double den = Math.Sqrt(p) * Math.Sqrt(q);
            return (sum == 0) ? 0 : sum / den;
        }

        //Function to extract text documents
        public static string GetTextDoc(string path)
        {
            Document document1 = new Document();
            document1.LoadFromFile(path);

            //Initialzie StringBuilder Instance
            StringBuilder sb = new StringBuilder();

            //Extract Text from Word and Save to StringBuilder Instance
            foreach (Spire.Doc.Section section in document1.Sections)
            {
                foreach (Spire.Doc.Documents.Paragraph paragraph in section.Paragraphs)

                {
                    sb.AppendLine(paragraph.Text);
                }
            }
            return sb.ToString();

        }
        //Function to extract PDF documents
        public static string GetTextPdf(string path)
        {
            Spire.Pdf.PdfDocument pdoc = new Spire.Pdf.PdfDocument();
            pdoc.LoadFromFile(path);

            //Initialzie StringBuilder Instance for pdf
            StringBuilder sbpdf = new StringBuilder();
            //Extract text from all pages
            foreach (Spire.Pdf.PdfPageBase page in pdoc.Pages)
            {
                sbpdf.Append(page.ExtractText());
            }

            return sbpdf.ToString();
        }

        //Function to extract job description
        public static string GetTxtText(string path)
        {
            string text = System.IO.File.ReadAllText(path);

            return text.ToString();
        }

        //Word embeddings text pipeline
        public static IEstimator<ITransformer> TextPipeline()
        {
            //create the transformation pipeline
            var textPipeline = mlContext.Transforms.Text.NormalizeText("Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens",
                    Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.English))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens",
                WordEmbeddingEstimator.PretrainedModelKind.GloVe50D));
            return textPipeline;
        }
    }
}
