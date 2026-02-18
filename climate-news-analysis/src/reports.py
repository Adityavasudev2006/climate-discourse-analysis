# src/reports.py

import pandas as pd

def generate_bias_report(analysis_results, output_path):

    print(f"Generating bias and event correlation report at {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Media Bias Analysis Report\n")
        f.write("==========================\n\n")
        f.write("This report analyzes sentiment bias by comparing each source's article sentiment\n")
        f.write("to the regional average for articles on the same topic published in the same month.\n")
        f.write("A positive bias score means the source was generally more positive than its regional peers.\n")
        f.write("A negative score means it was more negative.\n\n")

        #  Section 1: Overall Bias Rankings 
        f.write("--- Overall Bias Rankings by Source ---\n")
        f.write("---------------------------------------\n")
        
        # Sort sources from most positively biased to most negatively biased
        sorted_sources = sorted(analysis_results.items(), key=lambda item: item[1]['overall_bias_score'], reverse=True)
        
        for source, data in sorted_sources:
            f.write(f"- {source.upper()}: {data['overall_bias_score']:.4f}\n")
        f.write("\n")

        #  Section 2: Detailed Analysis per Source 
        f.write("--- Detailed Source Analysis ---\n")
        f.write("--------------------------------\n")
        
        for source, data in sorted_sources:
            f.write(f"### SOURCE: {source.upper()} ###\n")
            f.write(f"Overall Bias Score: {data['overall_bias_score']:.4f}\n\n")
            
            if not data['changepoints']:
                f.write("No significant shifts in sentiment bias detected during the analyzed period.\n")
            else:
                f.write("Detected shifts in sentiment bias and potential correlated events:\n")
                for cp in data['changepoints']:
                    change_date = cp['date']
                    event = cp['correlated_event']
                    
                    f.write(f"  - On/Around {change_date.strftime('%Y-%m-%d')}:\n")
                    f.write(f"    - A significant shift in reporting bias was detected.\n")
                    if event != 'None':
                        f.write(f"    - This shift occurred shortly after the '{event}' event.\n")
                    else:
                        f.write(f"    - No major predefined event was found to correlate with this shift.\n")
            
            f.write("\n--------------------------------\n\n")

    print("Bias report generation complete.")