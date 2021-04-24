# @author: Krishna Kariya, Fahad Nayyar, 2021

import os

def execute_command(cmd):
   print(cmd)
   print()
   os.system(cmd)
   return

pwd_str = os.getcwd()
if pwd_str[-1]!="/":
   pwd_str += "/"
sat_dir = pwd_str+"test/sat/"
unsat_dir = pwd_str+"test/unsat/"

# os.chdir(sat_dir)
for cnf_file in os.listdir(sat_dir):
   if cnf_file.endswith(".cnf"):
      cnf_file_path = sat_dir+cnf_file
      print("running preprocessing on : " + cnf_file_path)
      os.environ['MODE'] = str(0)
      cmd = pwd_str+"preprocessor " + cnf_file_path
      execute_command(cmd)
      os.environ['MODE'] = str(1)
      cmd = pwd_str+"preprocessor " + cnf_file_path
      execute_command(cmd)


      
for cnf_file in os.listdir(unsat_dir):
   if cnf_file.endswith(".cnf"):
      cnf_file_path = sat_dir+cnf_file
      print("running preprocessing on : " + cnf_file_path)
      os.environ['MODE'] = str(0)
      cmd = pwd_str+"preprocessor " + cnf_file_path
      execute_command(cmd)
      os.environ['MODE'] = str(1)
      cmd = pwd_str+"preprocessor " + cnf_file_path
      execute_command(cmd)
      
