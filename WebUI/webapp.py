#!/usr/bin/env python3
from  flask import *
import sqlite3
from functools import wraps
import hashlib
import logging
import os
import subprocess

OS_USER = subprocess.check_output(["whoami"], universal_newlines=True).splitlines()[0]
DATABASE = '/home/' + OS_USER + '/SmartSlam/DB/smartSlamDB.sqlite'
DATABASEUSERS = '/home/' + OS_USER + '/SmartSlam/WebUI/users.db'
app = Flask(__name__)
app.config.from_object(__name__)

app.secret_key='my_precious'

log_dir = '/home/' + OS_USER + '/SmartSlam/WebUI/LOG/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename='/home/' + OS_USER + '/SmartSlam/WebUI/LOG/INFO.log',
                    level=logging.INFO,
                    format='%(asctime)-15s '
                           '%(levelname)s '
                           '--%(filename)s-- '
                           '%(message)s')


def connect_db():
    return sqlite3.connect(app.config['DATABASE'])

def connect_usersdb():
    return sqlite3.connect(app.config['DATABASEUSERS'])

@app.route('/',methods=['GET','POST'])
def home():
    g.db = connect_usersdb()
    cur = g.db.execute('select username, password from users')
    users = [dict(username=row[0], password=row[1]) for row in cur.fetchall()]

    g.db.close()
    error = None

    if request.method == 'POST':
        username = request.form['username']
        password=request.form['password']
        for user in users:

            if user['username']==username and check_password(user['password'],password):
                session['logged_in'] = True
                return redirect(url_for('accesses'))

            else:
                error = 'Invalid Credentials. Please try again.'
    return render_template('log.html')

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

def login_required(test):
    @wraps(test)
    def wrap(*args,**kwargs):
        if 'logged_in' in session:
            return test(*args,**kwargs)
        else:
            flash('you need to login first.')
            return redirect(url_for('log'))
    return wrap


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('you were logged out')
    return redirect(url_for('log'))

@app.route('/accesses')
@login_required
def accesses():
    g.db=connect_db()
    cur = g.db.execute('select timestamp, name,label,path  from accesses')
    accesses=[dict(timestamp=row[0], name=row[1], label=row[2], path=row[3],present= os.path.isfile(row[3])) for row in cur.fetchall()]
    cur=g.db.execute('select label from labels')
    labels=[dict(label=row[0]) for row in cur.fetchall()]
    g.db.close()
    return render_template('accesses.html', accesses=accesses, labels=labels)

@app.route('/download/<string:name>')
@login_required
def download(name):
    g.db = connect_db()
    cur = g.db.execute('select path  from accesses where name=(?)',[name])
    path = [row[0] for row in cur.fetchall()]
    g.db.close()
    path=path[0]
    folders = path.split('/')
    filename = folders[len(folders) - 1]
    pathf = path[:(len(path) - len(filename) - 1)]
    print(pathf)
    print(filename)
    return send_from_directory(pathf,filename, as_attachment=True )

@app.route('/play/<string:name>')
@login_required
def play(name):
    g.db = connect_db()
    cur = g.db.execute('select path  from accesses where name=(?)', [name])
    path = [row[0] for row in cur.fetchall()]
    g.db.close()
    path = path[0]
    def generate():
        with open(path, 'rb') as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")


@app.route('/labels')
@login_required
def labels():
    g.db=connect_db()
    cur=g.db.execute('select label from labels')
    labels=[dict(label=row[0]) for row in cur.fetchall()]
    g.db.close()
    return render_template('labels.html', labels=labels)

@app.route('/update/<string:name>', methods=['GET','POST'])
@login_required
def update(name):
    lab = request.form['priority']
    if not lab:
        
        return redirect(url_for('accesses'))
    else:
        g.db = connect_db()
        cur = g.db.execute("UPDATE accesses SET label = (?) WHERE name = (?);", [lab, name])
        g.db.commit()
        g.db.close()
        
        return redirect(url_for('accesses'))



@app.route('/new_task', methods=['GET','POST'])
@login_required
def new_task():
    lab=request.form['name']
    if not lab :
       
        return redirect(url_for('labels'))
    else:
        g.db = connect_db()
        cur=g.db.execute("INSERT INTO labels (label) VALUES (?)", [lab] )
        g.db.commit()
        g.db.close()
        
        return redirect(url_for('labels'))

@app.route('/delete/<string:label>',)
@login_required
def delete_entry(label):
    g.db=connect_db()
    cur=g.db.execute('delete from labels where label=(?)',[label])
    g.db.commit()
    g.db.close()

    return redirect(url_for('labels'))


@app.route('/log', methods=['GET','POST'])
def log():
    g.db = connect_usersdb()
    cur = g.db.execute('select username, password from users')
    users = [dict(username=row[0], password=row[1]) for row in cur.fetchall()]

    g.db.close()
    error = None

    if request.method == 'POST':
        username = request.form['username']
        password=request.form['password']
        for user in users:

            if user['username']==username and check_password(user['password'],password):
                session['logged_in'] = True
                return redirect(url_for('accesses'))

            else:
                error = 'Invalid Credentials. Please try again.'

    return render_template('log.html', error=error)

def check_password(hashed_password, user_password):
    password, salt = hashed_password.split(':')
    return password == hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080, debug=True)
