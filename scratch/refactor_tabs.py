import sys

def modify_app():
    with open("app.py", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start_run_idx = 445 - 1  # 0-indexed
    start_hist_idx = 737 - 1
    
    # Find end of history block (the start of the Analytics page)
    end_hist_idx = -1
    for i in range(start_hist_idx + 1, len(lines)):
        if lines[i].startswith("elif page == "):
            end_hist_idx = i
            break
            
    if end_hist_idx == -1:
        end_hist_idx = len(lines)
            
    new_lines = lines[:start_run_idx]
    new_lines.append("    tab_run, tab_history = st.tabs([\"🚀 Run Evaluation\", \"📜 Evaluation History\"])\n")
    new_lines.append("\n    with tab_run:\n")
    
    # Indent run block
    for i in range(start_run_idx, start_hist_idx):
        if lines[i].strip() == "":
            new_lines.append("\n")
        else:
            new_lines.append("    " + lines[i])
            
    new_lines.append("\n    with tab_history:\n")
    # Indent history block
    for i in range(start_hist_idx, end_hist_idx):
        if lines[i].strip() == "":
            new_lines.append("\n")
        else:
            new_lines.append("    " + lines[i])
            
    # append the rest of the file
    new_lines.extend(lines[end_hist_idx:])
    
    with open("app.py", "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Modified successfully")

if __name__ == "__main__":
    modify_app()
