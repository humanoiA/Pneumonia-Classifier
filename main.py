from flask import Flask,abort,render_template,request,redirect,url_for,send_file,after_this_request
from werkzeug import secure_filename
import glob
import sys
import os
import model
app = Flask(__name__)
UPLOAD_FOLDER = os.getcwd()+'\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
@app.route('/',methods = ['GET','POST'])
def upload_file():
    forward_message=""
    uploaded_file="No file chosen."
    test = UPLOAD_FOLDER+'\\*'
    r = glob.glob(test)
    for i in r:
        os.remove(i)
    if request.method =='POST':
        file = request.files['file[]']
        if file:
            filename = secure_filename(file.filename)
            uploaded_file=filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            os.rename(UPLOAD_FOLDER+'\\'+filename,UPLOAD_FOLDER+'\\'+'test.jpg')
            forward_message = "Prediction output: "+model.predictor()
    return render_template('app.html', forward_message=forward_message,uploaded_file=uploaded_file);
if __name__ == '__main__':
    app.run(debug = True)