#%%
from multiprocessing import Process
from mysql.connector import Error
# import mysql.connector
# import pdb;

def getFromDB(con,table,fields,wherestr=''):
	colstr = ""
	cur=con.cursor()
	for k in fields:
		colstr+= "`" + k + "`" + ","
		
	colstr = colstr[:-1] + " "
	if len(wherestr)==0:
		sqlstr = "select " + colstr + "from " + table 
	else:	
		sqlstr = "select " + colstr + "from " + table + " where " + wherestr

	try:
		print sqlstr		
		cur.execute(sqlstr)
		return cur.fetchall()
	except Error as err:
                print("Something went wrong: {}".format(err))
		return 1


def saveToDB(con,table,mydict):

	def insert(cur,table,mydict):
		sqlstr = "insert into " + table
		colstr = " ("
		valstr = " values ("

		for k in mydict.keys():
			colstr+= "`" + k + "`" + ","
			valstr+=  "%s"  + ","		

		colstr = colstr[:-1] + ")"
		valstr = valstr[:-1] + ")"
		wholestr = sqlstr + colstr + valstr

		try:
			cur.execute(wholestr,(mydict.values()))
			return 0
		
                except Error as err:
                        print("Something went wrong: {}".format(err))
			print wholestr
			return 1

	cur=con.cursor()
	if isinstance(mydict,dict):
		err=insert(cur,table,mydict)
	else:
		# pdb.set_trace()
		try:
			for d in mydict:
				err=insert(cur,table,d)
				if err==1:
					break
		except:
			print "mydict may be of the wrong type. Must be a dictionary or a sequence/list/generator of dictionaries"
			err=1
     
	if (err == 0):
		con.commit()
	else:
		con.rollback()
		print 'Error saving to DB'

	cur.close()
	return err


def saveToDB_m(con,table,mydict):
	p = Process(target=saveToDB_na, args=(con,table,mydict))
	#t.daemon=True
	p.start()
	return p


#%%
if __name__ == '__main__':

#%%
	import mysql.connector as db
	import numpy as np
#%%
	con=db.connect(host='erlichfs',user='bo', password='mayr2000',database='ann')
	
#%%	
	ass = int(raw_input('array size? '))
	comment = raw_input('comment? ')

	
	w = np.random.rand(ass)
	wd = w.dumps()
        #runid=1
	# t=saveToDB(con,'ann.test2',
        #            ({'id':runid,'data':wd, 'text':comment},
        #             {'id':runid+1,'data':1234324, 'text':'foo'}))	
	t=saveToDB(con,'ann.test2',
                   ({'data':wd},
                    {'text':'foo'}))	
	t.join()    
	# We don't want to kill the mysql connection until all the threads are done!
	con.close()
	
	#print 'dict saved'
	#
	#saveToDB(con,'ann.test2',[{'data':wd},{'data':wm}])
	#print 'tuple of dicts saved'
	
	
	#%%
