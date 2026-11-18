import { NavLink } from "react-router-dom";

export function Navigation() {
  return <header className="topbar"><strong className="project-name">AI Research Assistant</strong><nav aria-label="Primary navigation"><NavLink to="/">Research Assistant</NavLink><NavLink to="/dashboard">Evaluation Dashboard</NavLink></nav></header>;
}
