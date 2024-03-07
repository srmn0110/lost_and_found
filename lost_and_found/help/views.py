from django.shortcuts import render
import time
# Create your views here.

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")


from transformers import ViltProcessor, ViltForQuestionAnswering #for a different model
processor_1 = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model_1 = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")




def index(request):
    a = time.time()
    if request.method == 'POST':
        # Code to handle POST request
        print(request.POST)
        text = request.POST.get('description')
        print(text)
        l = []

        # from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

        # model_name = "deepset/roberta-base-squad2"

        # # a) Get predictions
        # nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        QA_input = {
            'question': 'what object did i lose?',
            'context': text
        }
        res = nlp(QA_input)

        # b) Load model & tokenizer
        # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("\n\t the user lost a ",res['answer'] , "\n")

        
        import os
        from PIL import Image
        import requests


        images_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'images')
        images_folder_1 = os.path.join( 'static', 'images')
        # from transformers import BlipProcessor, BlipForQuestionAnswering

        # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        # model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

        # Loop through the images in the folder
        for filename in os.listdir(images_folder):
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or True:
                image_path = os.path.join(images_folder, filename)
                image = Image.open(image_path)
                
                image_path_1 = os.path.join(images_folder_1, filename)


                # prepare image + question
                text = f"is there a {res['answer']}"
                
                # processor_1 = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
                # model_1 = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

                # prepare inputs
                encoding = processor_1(image, text, return_tensors="pt")

                # forward pass
                outputs = model_1(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                
                y= model_1.config.id2label[idx]
                

                
                question = f"is there a {res['answer']} "
                # print("\n\tquestion = ",question)
                print("\n ")
                inputs = processor(image, question, return_tensors="pt")

                out = model.generate(**inputs)
                # print(processor.decode(out[0], skip_special_tokens=True))

                x = processor.decode(out[0], skip_special_tokens=True)
                print("\n\tvilt == ",y,"\tblip based = ",x,"\t",filename)
                if x == "yes" or y == "yes":
                    l.append(image_path_1[7:])


        context = {'image_list': l,'time_taken':time.time()-a}
        print("\n\tl",l,"\n")
        print("\ttime taken = ",time.time()-a,"\n")
        return render(request, 'help/index.html', context)

        





    return render(request, 'help/index.html')



