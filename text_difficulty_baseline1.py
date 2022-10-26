import numpy as np 
import js2py
from data_processing import train_data, test_data

# THIS CODE DOESN"T DO ANYTHING RN

def get_cefr_levels():
    # test_strings = test_data()[:, 14]
    test_string = 'A Black Widow is a shiny black spider. It has an orange or red mark that looks like an hourglass. Its abdomen is shaped like a sphere and has an hourglass mark on the bottom. Often there are just two red marks separated by black. Females sometimes have the hourglass shape on top of the abdomen above the silk-spinning organs (spinnerets). Females are usually about 1-1/2 inches long including their leg span. In areas where grapes grow, females are very small and round. They resemble shiny black or red grapes.\nMale Black Widows are much smaller than females. Their bodies are only about 1/4 inch long. They can be either gray or black. They do not have an hourglass mark, but may have red spots on the abdomen.\nBlack widows are sometimes called “comb-footed” spiders. The bristles on their hind legs are used to cover trapped prey with silk.\nYoung spiders are called “spiderlings”. They shed their outer covering (exoskeleton) as they grow. Spiderlings are orange, brown, or white at first and get darker each time they shed their skin (molt).'
    # js2py.translate_file("cefr_bot.js", "temp.py")
    
    eval_result, temp = js2py.run_file('cefr_bot.js')
    temp.run_cefr_bot(test_string)


if __name__ == "__main__":
    get_cefr_levels()
