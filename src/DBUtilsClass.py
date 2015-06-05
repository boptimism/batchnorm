


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
		saveSettings	saves your connection information to .dbconf in your home directory
		use 			set the default database

	"""
	

	def __init__(self, *args):
		""" con = Connection() looks for a .dbconf file in your home directory with 
			con = Connection(host,user,passwd,db='')
			con = Connection(dbconf)
				where dbconf is a dictionary with user,host,passwd, database and other parameters to pass to the connect argument of the module. See http://mysql-python.sourceforge.net/MySQLdb.html#functions-and-attributes for a full list of possible connection attributes."""

		
		# Try to use the native MySQLdb module if possible.
		try:
			import MySQLdb as db
			print "Using MySQLdb"
		except ImportError:
			print "MySQLdb not found"
			try:				
				import mysql.connector as db
				from mysql.connector import Error
			except ImportError:
				print "Neither MySQLdb nor mysql.connector found.  Dependency failure"
				raise ImportError('Cannot find MySQLdb or mysql.connector module')

		import ConfigParser as cp
		self.config = cp.ConfigParser()

		# Parse the inputs		
		if len(args)==0:   # Try to use defaults in ~/.dbconf
			import ConfigParser as cp
			self.config.read(self.defaultConfigFile())   # This will throw an error in some cases. Like running on windows 
			args = [None]
			args[0] = dict((x, y) for x, y in config.items('client'))
			print "Loaded defaults from ~/.dbconf"
		

		if len(args)==1 and isinstance(args[0],dict):
			self.cfg = args[0]
			self.con=db.connect(**self.cfg)
		elif len(args)==1:
			raise Exception('When calling Connection with ')	
		else:
			l = ('host','user','passwd','db')
			try:
				for x,y in zip(l,args):
					d[x]=y
			except IndexError:  # We might run out of arguments, since db is optional
				pass				

			self.con=db.connect(**d)


	def __del__(self):
		pass
		# If you use multithreading or processing you must make sure everything is done before closing up!

	def checkConnection(self):
		pass

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


	def lastInsertID(self):
		return cur.fetchone()[0]


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

	def execute(con,sqlstr,closeit=False):
		try:
			cur=con.cursor()
			cur.execute(sqlstr)
			con.commit()
			cur.close()
		except: 
			'put in proper mysql error'
			con.rollback()

		if closeit:
			con.close()
			print 'closed connection'




	def execute_m(con,sqlstr):
		p = Process(target=execute, args=(con,sqlstr,True))
		p.start()


if __name__ == "__main__":
	import getpass
	d = {'host':'erlichfs', 'user':'jerlich', 'passwd':getpass.getpass()}
	c = Connection(d)
	c.saveSettings()


