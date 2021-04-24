# @author: Krishna Kariya, Fahad Nayyar, 2021

import os

pwd_str = os.getcwd()
if pwd_str[-1]!="/":
   pwd_str += "/"
sat_dir = pwd_str+"tests/sat/"
unsat_dir = pwd_str+"tests/unsat/"
speedup_result_file_path = pwd_str+"speedup_results.txt"

def execute_command(cmd):
   speedup_result_fp = open(speedup_result_file_path,'a+')
   speedup_result_fp.write("\n")
   speedup_result_fp.write("COMMAND: " + cmd)
   speedup_result_fp.close()
   os.system(cmd)
   return

cmd = ": > " + speedup_result_file_path
execute_command(cmd)

for cnf_file in os.listdir(sat_dir):
   if cnf_file.endswith(".cnf"):
      cnf_file_path = sat_dir+cnf_file
      os.environ['MODE'] = str(0)
      cmd = "timeout 10 " + pwd_str+"preprocessor " + cnf_file_path + " >> " + speedup_result_file_path
      execute_command(cmd)
      os.environ['MODE'] = str(1)
      cmd = "timeout 10 " + pwd_str+"preprocessor " + cnf_file_path + " >> " + speedup_result_file_path
      execute_command(cmd)


      
for cnf_file in os.listdir(unsat_dir):
   if cnf_file.endswith(".cnf"):
      cnf_file_path = sat_dir+cnf_file
      os.environ['MODE'] = str(0)
      cmd = "timeout 10 " + pwd_str+"preprocessor " + cnf_file_path + " >> " + speedup_result_file_path
      execute_command(cmd)
      os.environ['MODE'] = str(1)
      cmd = "timeout 10 " + pwd_str+"preprocessor " + cnf_file_path + " >> " + speedup_result_file_path
      execute_command(cmd)
      
