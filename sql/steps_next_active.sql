SELECT id, step_order, description FROM steps WHERE goal_id = ? AND status = 'active' ORDER BY step_order LIMIT 1
