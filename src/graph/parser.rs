//! Plain-text graph parser.
//!
//! Format:
//!   * first non-empty line: `n m`  (number of vertices, number of edges)
//!   * then exactly `m` lines: `u v` (0-based vertex numbers)
//!   * blank lines and lines starting with `#` are ignored
//!
//! Author: <Hasan>  Login: <xhasan00>

use super::{Edge, Graph, Vertex, VertexCount};
use std::{fmt, fs, path::Path};

/// Error checker while reading a graph file.
#[derive(Debug, PartialEq, Eq)]
pub enum ParseError {
    Io(String),
    MissingHeader,
    InvalidNumber { line: usize, token: String },
    BadEdgeLine { line: usize },
    TooManyVertices(u64),
    VertexOutOfRange { line: usize, vertex: u64, num_vertices: u64 },
    EdgeCountMismatch { expected: usize, found: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Io(e) => write!(f, "cannot read file: {e}"),
            ParseError::MissingHeader => write!(f, "missing header line `n m`"),
            ParseError::InvalidNumber { line, token } => {
                write!(f, "line {line}: `{token}` is not a number")
            }
            ParseError::BadEdgeLine { line } => {
                write!(f, "line {line}: expected exactly two vertices `u v`")
            }
            ParseError::TooManyVertices(n) => write!(f, "{n} vertices do not fit in 32 bits"),
            ParseError::VertexOutOfRange { line, vertex, num_vertices } => {
                write!(f, "line {line}: vertex {vertex} is outside 0..{num_vertices}")
            }
            ParseError::EdgeCountMismatch { expected, found } => {
                write!(f, "header promises {expected} edges but the file has {found}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

fn number(token: &str, line: usize) -> Result<u64, ParseError> {
    token.parse::<u64>().map_err(|_| ParseError::InvalidNumber {
        line,
        token: token.to_string(),
    })
}

/// Parses a graph from text.
///
/// Every vertex number is checked before `Graph::new` is called, because
/// `Graph::new` panics on an out-of-range endpoint and a parser should return
/// an error instead.
pub fn parse_graph(text: &str, is_directed: bool) -> Result<Graph, ParseError> {
    // (line number, trimmed text), skipping blank lines and comments.
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'));

    let (header_line, header) = lines.next().ok_or(ParseError::MissingHeader)?;
    let mut parts = header.split_whitespace();
    let n = number(parts.next().ok_or(ParseError::MissingHeader)?, header_line)?;
    let m = number(parts.next().ok_or(ParseError::MissingHeader)?, header_line)? as usize;
    if n > VertexCount::MAX as u64 {
        return Err(ParseError::TooManyVertices(n));
    }

    let mut edges: Vec<Edge> = Vec::with_capacity(m);
    for (line, text) in lines {
        let mut p = text.split_whitespace();
        let (a, b) = match (p.next(), p.next(), p.next()) {
            (Some(a), Some(b), None) => (a, b),
            _ => return Err(ParseError::BadEdgeLine { line }),
        };
        let (u, v) = (number(a, line)?, number(b, line)?);
        for x in [u, v] {
            if x >= n {
                return Err(ParseError::VertexOutOfRange { line, vertex: x, num_vertices: n });
            }
        }
        edges.push((u as Vertex, v as Vertex));
    }

    if edges.len() != m {
        return Err(ParseError::EdgeCountMismatch { expected: m, found: edges.len() });
    }
    Ok(Graph::new(n as VertexCount, &edges, is_directed))
}

/// Reads and parses a graph file.
pub fn read_graph_file(path: impl AsRef<Path>, is_directed: bool) -> Result<Graph, ParseError> {
    let text = fs::read_to_string(path).map_err(|e| ParseError::Io(e.to_string()))?;
    parse_graph(&text, is_directed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_triangle() {
        let g = parse_graph("3 3\n0 1\n1 2\n2 0\n", false).unwrap();
        assert_eq!(g.num_vertices, 3);
        assert_eq!(g.out_degree(0), 2);
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        assert!(parse_graph("# my graph\n\n2 1\n  0   1  \n\n", false).is_ok());
    }

    #[test]
    fn accepts_zero_edges() {
        assert_eq!(parse_graph("4 0\n", false).unwrap().num_vertices, 4);
    }

    #[test]
    fn rejects_empty_or_headerless_input() {
        assert_eq!(parse_graph("", false).unwrap_err(), ParseError::MissingHeader);
        assert_eq!(parse_graph("3\n", false).unwrap_err(), ParseError::MissingHeader);
    }

    #[test]
    fn rejects_bad_number() {
        assert!(matches!(
            parse_graph("3 1\n0 x\n", false),
            Err(ParseError::InvalidNumber { line: 2, .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_vertex() {
        assert!(matches!(
            parse_graph("3 1\n0 3\n", false),
            Err(ParseError::VertexOutOfRange { vertex: 3, .. })
        ));
    }

    #[test]
    fn rejects_wrong_edge_count() {
        assert_eq!(
            parse_graph("3 2\n0 1\n", false).unwrap_err(),
            ParseError::EdgeCountMismatch { expected: 2, found: 1 }
        );
    }

    #[test]
    fn rejects_malformed_edge_line() {
        assert!(matches!(
            parse_graph("3 1\n0 1 2\n", false),
            Err(ParseError::BadEdgeLine { line: 2 })
        ));
    }

    #[test]
    fn missing_file_is_io_error() {
        assert!(matches!(
            read_graph_file("/no/such/file.txt", false),
            Err(ParseError::Io(_))
        ));
    }
}   