
from flask import Flask , render_template , request, redirect
import test


app = Flask(__name__)

@app.route("/")

@app.route("/index" , methods=["GET", "POST"])
def index():
    if request.method == "POST":

        req = request.form
        print (req)
        missing = list()

        for k, v in req.items():
            if v == "":
                missing.append(k)

        if missing:
            feedback = f"Missing fields for:   {',   '.join(missing)}"
            return render_template("index.html", feedback=feedback)
        
        else:
            
            x = list()

            for k, v in req.items():
                try:    
                    x.append(float(v))
                except ValueError:
                    feedback = f"The value of {k} must be a number"
                    return render_template("index.html", feedback=feedback)
            
            feedback = "foll"
            print(x)
            
            request.url += "?res="+str(test.predict([x]))
            return redirect(request.url)
            

        return redirect(request.url)
    return render_template('index.html')


