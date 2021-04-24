# @author: Krishna Kariya, Fahad Nayyar, 2021

import os
import sys
import time

pwd_str = os.getcwd()
if pwd_str[-1]!="/":
   pwd_str += "/"
sat_dir = pwd_str+"tests/sat/"
sat_large_dir = pwd_str+"tests_large/sat/"
unsat_dir = pwd_str+"tests/unsat/"
unsat_large_dir = pwd_str+"tests_large/unsat/"
edusat_executable_path = pwd_str+"Edusat/edusat"
output_file_path =  pwd_str+"out8.txt"
removed_var_cnf_path = pwd_str+"removed_vars.cnf"
temp_file_path = pwd_str+"temp.txt"

def execute_command(cmd):
   
   # temp_file_fp = open(temp_file_path, 'a+')
   # temp_file_fp.write("COMMAND: " + cmd + "\n")
   # temp_file_fp.close()
   print("COMMAND: " + cmd )
   os.system(cmd)
   return


passed=1
for cnf_file in os.listdir(sat_dir):
   if cnf_file.endswith(".cnf"):
      cnf_file_path = sat_dir+cnf_file
      cmd = ": > " + removed_var_cnf_path
      os.environ['MODE'] = str(1)
      cmd = "timeout 10 " + pwd_str+"preprocessor " + cnf_file_path + " >> " + temp_file_path
      execute_command(cmd)
      cmd = "timeout 10 " + edusat_executable_path + " " + removed_var_cnf_path + " >> " + temp_file_path
      execute_command(cmd)
      temp_file_fp = open(temp_file_path, 'r')
      lines = temp_file_fp.readlines()
      if (lines[-1].find("SAT")==-1):
         print("FAILED: on cnf file: " + cnf_file_path)
         passed=0
      time.sleep(2)
if passed==1:
   print("PASSED on all test cases")


