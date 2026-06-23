# clean_scenarios.py

import json
import re

def clean_scenarios(file_path):
    """Clean encoding issues and fix JSON syntax errors"""
    
    print(f"📂 Reading file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📏 Original file size: {len(content)} characters")
    
    # Fix all encoding issues
    fixes = [
        (r'â†’', '→'),
        (r'â€“', '–'),
        (r'â€™', "'"),
        (r'â€"', '"'),
        (r'â€œ', '"'),
        (r'â€\u009d', '"'),
        (r'â€\u009c', '"'),
        (r'Â', ''),
        (r'\u2013', '–'),
        (r'\u2014', '—'),
        (r'\u2018', "'"),
        (r'\u2019', "'"),
        (r'\u201c', '"'),
        (r'\u201d', '"'),
    ]
    
    print("🔧 Applying encoding fixes...")
    for old, new in fixes:
        content = content.replace(old, new)
    
    # Try to find the error location
    print("\n🔍 Attempting to parse JSON...")
    try:
        data = json.loads(content)
        print("✅ JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error at position {e.pos}: {e.msg}")
        print(f"   Line {e.lineno}, Column {e.colno}")
        
        # Show the problematic section
        start = max(0, e.pos - 100)
        end = min(len(content), e.pos + 100)
        snippet = content[start:end]
        print(f"\n📄 Problematic section:")
        print("-" * 60)
        print(snippet)
        print("-" * 60)
        print(" " * 100 + "⬆️ Error here")
        
        # Try to fix common JSON issues
        print("\n🔧 Attempting automatic fixes...")
        
        # Fix trailing commas in objects and arrays
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Fix missing commas between array elements
        # (Find pattern: } { or ] [ or " " without comma)
        content = re.sub(r'}\s*{', '}, {', content)
        content = re.sub(r']\s*\[', '], [', content)
        content = re.sub(r'"\s*"', '", "', content)
        
        # Try parsing again
        try:
            data = json.loads(content)
            print("✅ Automatic fixes succeeded!")
        except json.JSONDecodeError as e2:
            print(f"❌ Automatic fixes failed: {e2}")
            print("\n💡 Manual intervention required.")
            print(f"   Check line {e.lineno} in the file.")
            return None
    
    # Save fixed file
    print("\n💾 Saving fixed file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Fixed encoding in {file_path}")
    print(f"📏 New file size: {len(json.dumps(data, indent=2))} characters")
    
    # Also save a backup
    backup_path = file_path.replace('.json', '_backup.json')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Backup saved to: {backup_path}")
    
    return data

def inspect_problematic_line(file_path, line_num):
    """Helper to inspect a specific line in the file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if line_num <= len(lines):
        print(f"\n📄 Line {line_num}:")
        print("-" * 60)
        print(lines[line_num - 1])
        print("-" * 60)
        
        # Show surrounding lines for context
        start = max(0, line_num - 3)
        end = min(len(lines), line_num + 3)
        print(f"\n📄 Context (lines {start+1}-{end}):")
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            print(f"{prefix}{i+1}: {lines[i].rstrip()}")

if __name__ == "__main__":
    file_path = 'data/json/scenarios.json'
    
    # First attempt with automatic fixes
    result = clean_scenarios(file_path)
    
    if result is None:
        print("\n❌ Could not fix automatically.")
        print("Please check the error location and manually fix the JSON.")
        print("\n🔧 To inspect line 1762 specifically:")
        inspect_problematic_line(file_path, 1762)