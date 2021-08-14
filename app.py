import os

from flask import Flask, render_template, request, redirect

from inference import get_prediction, sigmoid, to_figure
from commons import format_class_name, transform_image

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # prédiction sur l'image
        import os
        import onnxruntime
        import onnx
        from onnx2pytorch import ConvertModel
        import pandas as pd
        model_path = os.path.join('models', 'MedNet.onnx')
        ort_session = onnxruntime.InferenceSession(model_path)
        onnx_model = onnx.load(model_path)
        pytorch_model = ConvertModel(onnx_model)

        """
        import sys
        print("position du fichier :",os.path.dirname(sys.argv[0]))
        # importation du nom de fichier
        if 'files' not in request.files:
            print("redirection")
            return redirect(request.url)
        file = request.files.get('files')
        #print(request.files["files"])
        #print(dir(request.files))
        #print("headers\n:", file.headers ) 
        #print("stream\n:", file.stream ) 
        print("file\n:", file.filename ) 
        chemin, picture = file.filename.split("/")[:-1], file.filename.split("/")[-1:]
        print(chemin, picture)
        dataDir = "/".join(chemin)              # The main data directory
        classNames = os.listdir(dataDir)  # Each type of image can be found in its own subdirectory
        #numClass = len(classNames)        # Number of types = number of subdirectories
        #imageFiles = [[os.path.join(dataDir,classNames[i],x) for x in os.listdir(os.path.join(dataDir,classNames[i]))]
        #      for i in range(numClass)] 
        print("ClassNames : \n", classNames)  
        #print("DIR\n:", dir(file) , end="\n\n\n") 
        """
        """
        if not file: return
        #dataDir = 'to_predict'
        #import io
        #print("file\n:", dir(file) )             
        #classNames = file.read() #os.listdir(dataDir)  
        #print("classNames : \n",classNames)
        #imageFiles = [os.path.join(dataDir,name) for name in classNames] 
        #print('Images files\n', imageFiles) 
        """
        # importation des l'images
        if 'files' not in request.files:
            print("redirection")
            return redirect(request.url)
        files = request.files.getlist('files')
        df=pd.DataFrame(columns=['Fichier','Prediction'])
        datas = pd.DataFrame()
        for file in files:
            if not file: return
            img_bytes = file.read()
            class_name ,class_id,values = get_prediction(image_bytes=img_bytes, ort_session= ort_session)
            #directory_name = file.filename.split("/")[:-2]
            file_name = file.filename.split("/")[-1][:-5] # suppr .jpeg
            df = df.append({"Fichier":file_name, "Prediction":class_name}, ignore_index=True)
            datas[file_name]=list(values)
        import pickle
        with open('results.json', 'wb') as fp:
            pickle.dump(datas, fp)

        #FINDINGS = ["AbdomenCT","BreastMRI","ChestCT","CXR","Hand","HeadCT"]
        #min_results = min(values)
        #results = {k:v for k,v in zip(FINDINGS,values)}
        #to_figure(results)

        return html(df.to_html())
        #return render_template('result.html', class_id=class_id,
        #                       class_name=class_name, file_name=file.filename, 
        #                       aff1=df.to_html())

        """
        try: aff1 = ""
        except: aff1 = ""
        aff2=f"Les données d'{ort_session.get_inputs()[0].name} ont pour taille {ort_session.get_inputs()[0].shape[1:]}. "
        aff3=f"Les données d'{ort_session.get_outputs()[0].name} ont pour taille {ort_session.get_outputs()[0].shape[1:]}."
        aff4="results"
        aff5="" #results_mod
        aff6=""
        aff7=str(file.filename)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name, file_name=file.filename, 
                               aff1=aff1, 
                               aff2=aff2,
                               aff3=aff3, 
                               aff4=aff4, 
                               aff5=aff5, 
                               aff6=aff6, 
                               aff7=aff7)
        """
    return render_template('index.html')

def html(content):  # Also allows you to set your own <head></head> etc
   return '''<!doctype html><html lang="en">
   <head>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="//stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css" integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS" crossorigin="anonymous">
    <style>.bd-placeholder-img {font-size: 1.125rem;text-anchor: middle;}
        @media (min-width: 768px) {.bd-placeholder-img-lg {font-size: 3.5rem;}}</style>
    <style>table { margin: auto; }</style>
    <link rel="stylesheet" href="/static/style.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    <script>
         $(document).ready(function(){
             $('#show').click(function() {
               $('.menu').toggle("slide");
             });
         });
         $(document).ready(function(){
             $('.menu').toggle("slide")
         });
         $(document).ready(function(){
             $('.see_img').toggle("slide")
         });
         $(document).ready(function(){
             $('#see_pict').click(function() {
               $('.see_img').toggle("slide");
             });
         });
    </script>
    <title>Image Prediction using PyTorch</title>
   </head>
   <body class="text-center">
    <form class="form-signin" method=post enctype=multipart/form-data>
        <img class="mb-4" src="/static/logo.png" alt="" width="72">
        <h1 class="h3 mb-3 font-weight-normal" id="show">Prédictions</h1>                
        <div class="menu class="mt-5 mb-3 text-muted" style="display: none;">''' + content + '''</div>  
    </form>            
    <div style="width:100%;">
        <h1 class="h3 mb-3 font-weight-normal" id="see_pict" style="padding: 100px 0 0 0">Visualisation d'une prédiction</h1> 
		<input type="text"  Name="name" MAXLENGTH="6" ><br>
        <img class="see_img" src="static\image_results.png">
        <p class="see_img"> exemple de graphe pouvant apparaître </p>
    </div>
    <script src="//code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>            
    <script src="//cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.6/umd/popper.min.js" integrity="sha384-wHAiFfRlMFy6i5SRaxvfOCifBUQy1xHdJ/yoi7FRNXMRBu5WHdZYu1hA6ZOblgut" crossorigin="anonymous"></script>            
    <script src="//stackpath.bootstrapcdn.com/bootstrap/4.2.1/js/bootstrap.min.js" integrity="sha384-B0UglyR+jN6CkvvICOB2joaf5I4l3gm9GU6Hc1og6Ls7i6U/mkkaduKaBhlAXv9k" crossorigin="anonymous"></script>                      
   </body>        
   </html>'''


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
