# GitLab Wiki Pages

Place your wiki page markdown files (`.md`) in this directory.

## How It Works

Each `.md` file in this directory will be imported as a wiki page in the Documentation project (Project ID 1).

## File Naming

The filename (without `.md` extension) becomes the wiki page title:
- `home.md` → "home" page
- `Getting-Started.md` → "Getting-Started" page
- `API-Documentation.md` → "API-Documentation" page

## Format

Wiki pages should be standard Markdown:

```markdown
# Page Title

Some content here.

## Section 1

More content...

## Section 2

- Bullet point 1
- Bullet point 2

[Link to other page](other-page)
```

## Example Structure

```
wikis/
├── README.md                   ← This file (not imported)
├── home.md                     ← Main wiki page
├── getting-started.md          ← Getting started guide
├── development-guide.md        ← Development documentation
└── api-reference.md            ← API documentation
```

## Sample Wiki Page

Create `wikis/home.md`:

```markdown
# Welcome to EnterpriseLab

This is the company-wide documentation hub.

## Quick Links

- [Getting Started](getting-started)
- [Development Guide](development-guide)
- [API Reference](api-reference)

## About

EnterpriseLab is an enterprise environment with:
- GitLab for code management
- OwnCloud for file storage
- RocketChat for communication
- And more!
```

## During Build

The `init.sh` script will:
1. Find all `.md` files in this directory (except README.md)
2. Extract the title from filename
3. Create wiki pages in the Documentation project
4. Upload content to GitLab

## Accessing Wikis

After building and starting the container:
1. Go to http://localhost:8080
2. Login as root
3. Navigate to Documentation project
4. Click "Wiki" in the sidebar
5. Your pages will be there!

## Notes

- Only `.md` files are imported
- README.md files are typically skipped
- Use standard Markdown syntax
- Internal wiki links: `[text](page-name)` (no .md extension)
- The build process will show each wiki page being uploaded
