try:
	# Try to use the native MySQLdb module if possible.	
	import MySQLdb as db
	print "Using MySQLdb"
except ImportError:
	print "MySQLdb not found"
	try:				
		import mysql.connector as db
		from mysql.connector import Error
		print "Using mysql.connector"
	except ImportError:
		print "Neither MySQLdb nor mysql.connector found.  Dependency failure"
		raise ImportError('Cannot find MySQLdb or mysql.connector module')


from multiprocessing import Process
import sys

class Connection():
	"""This class wraps communication with a MySQL server. It will try to use MySQLdb as the module for connection, but will fallback to mysql.connector if necessary.

		Properties:
		con 	The database connection.
		cur 	A database cursor

		Functions:
		Connection 		creates an instance of the class.
		getFromDB		a wrapper for select statements, returns the tuple from cursor.fetchall()
		saveToDB 		a wrapper for insert statements, returns last_insert_id
		call			a wrapper for calling mysql procedures
		execute 		passes through to cur.execute for things not covered by the wrapper functions
		query 			An SQL query that returns the results
		saveSettings	saves your connection information to .dbconf in your home directory
		use 			set the default database

	"""
	

	def __init__(self, *args):
		""" con = Connection() looks for a .dbconf file in your home directory with 
			con = Connection(host,user,passwd,db='')
			con = Connection(dbconf)
				where dbconf is a dictionary with user,host,passwd, database and other parameters to pass to the connect argument of the module. See http://mysql-python.sourceforge.net/MySQLdb.html#functions-and-attributes for a full list of possible connection attributes."""

		import ConfigParser as cp
		self.config = cp.ConfigParser()
		try: 
			self.config.add_section('client')
		except cp.DuplicateSectionError:
			pass
			# It's ok if the section exists.

		# Parse the inputs		
		if len(args)==0:   # Try to use defaults in ~/.dbconf
			import ConfigParser as cp
			self.config.read(self.defaultConfigFile()) 
			args = [None]
			args[0] = dict((x, y) for x, y in self.config.items('client'))
			print "Loaded defaults from ~/.dbconf"
			# We should maybe check here that there is enough info to connect

		if len(args)==1 and isinstance(args[0],dict):
			self.cfg = args[0]
			for k,v in zip(self.cfg.keys(), self.cfg.values()):
				self.config.set('client',k,v)

		elif len(args)==1:
			raise Exception('When calling Connection with one argument it must be a dictionary')	
		else:
			l = ('host','user','passwd','db')
			try:
				d={}
				for x,y in zip(l,args):
					d[x]=y
					self.config.set('client',x,y)

			except IndexError:  # We might run out of arguments, since db is optional
				pass	

			self.cfg = d			

		self.checkConnection()


	def __del__(self):
		self.con.close()
		# If you use multithreading or processing you must make sure everything is done before closing up!

	def checkConnection(self):
		"""checkConnection()
				This function checks for a database connection and tries to get a new one if an open connection does not exist
		"""
		try:
			if self.con.open:
				pass
				#everything is good
			else:
				self.con = db.connect(**self.cfg)
				self.cur = self.con.cursor()
		except AttributeError:
			# This is the first time we are calling the function
			self.con = db.connect(**self.cfg)
			self.cur = self.con.cursor()


	def saveSettings(self):
		import os, stat
		fname = self.defaultConfigFile()
		fid = open(fname,'w')
		os.fchmod(fid.fileno(),stat.S_IWRITE + stat.S_IREAD)  # Make the file rw only by owner 

		self.config.write(fid)
		fid.close()

	def defaultConfigFile(self):
		import os
		from os.path import expanduser, sep
		return expanduser("~") + sep + ".dbconf"

		

	def getFromDB(self,table,fields,wherestr=''):
		colstr = ""
		for k in fields:
			colstr+= "`" + k + "`" + ","
			
		colstr = colstr[:-1] + " "
		if len(wherestr)==0:
			sqlstr = "select " + colstr + "from " + table 
		else:	
			sqlstr = "select " + colstr + "from " + table + " where " + wherestr

		try:
			self.cur.execute(sqlstr)
			return self.cur.fetchall()
		except Error as err:
			print("Something went wrong: {}".format(err))
			raise err


	def lastInsertID(self):
		self.cur.execute('select last_insert_id()')
		return cur.fetchone()[0]


	def saveManyToDB(self,table,cols,values):
		'''saveManyToDB(table,cols,vals)

			Table 	The table to insert into
			cols 	The column names to insert
			vals    a tuple of tuples.  Each inner tuple should have as many elements as cols

			e.g. a = ((1,2,3),(4,5,6))
			c.saveManyToDB('foo',('col1','col2','col3'),a)		
			'''

			sqlstr = "INSERT INTO " + table + " " + str(cols) + " VALUES " + str(values)[1:-1]
			self.cur.execute(sqlstr)
			self.con.commit()



	def saveToDB(self,table,mydict,closeit=False):

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
		if closeit:
			con.close()
			print 'closed connection'

		return err

	def saveToDB_m(con,table,mydict):
		p = Process(target=saveToDB, args=(con,table,mydict,True))
		p.start()
		return p

	def execute(self,sqlstr,closeit=False):
		'''Connection.execute(sql_command)
			use this for commands that do not require any return values.
		'''
		try:
			cur=self.con
			cur.execute(sqlstr)
			con.commit()
			cur.close()
		except: 
			'put in proper mysql error'
			con.rollback()

		if closeit:
			con.close()
			print 'closed connection'


	def query(self,sqlstr):
		'''Connection.query(sql_command)
			returns cursor.fetchall()
			Use this for commands that do require return values.
		'''

		try:
			cur=self.cur
			cur.execute(sqlstr)
			return cur.fetchall()
		except: 
			print 'Exception in query'


	def execute_m(con,sqlstr):
		p = Process(target=execute, args=(con,sqlstr,True))
		p.start()


if __name__ == "__main__":
	d = {'host':'erlichfs','user':'jerlich','passwd':'jce!u4$'}
	a = Connection(d)

	c = Connection()
	q=c.query('show schemas')
	for item in q:
		print item[0]
		tab = c.query('show tables from ' + item[0])
		for t in tab:
			print item[0] + '.' +  t[0]

	w = a.query('show schemas')
	print w




