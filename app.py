from flask import Flask,request, jsonify, render_template,redirect
import pickle as pkl

app = Flask(__name__)

cv_transformer = pkl.load(open("models/cv_transformer.pkl","rb"))
nb_model = pkl.load(open("models/nb_model.pkl","rb"))

def predict(text):
	wordvec = cv_transformer.transform([text])
	return nb_model.predict(wordvec)[0]

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/detect',methods=['GET','POST'])
def detect():
	if request.method == "POST":
		input_text = request.form["input_text"]
		p = predict(input_text)
		if(p==1):
			msg = "The given mail text seems to be Spam Mail..!!"
		else:
			msg = "The given mail text is not a Spam.."
		return render_template('index.html',p=p,msg=msg)
	return redirect('/')

if __name__ == '__main__':
	app.run(debug=True)
