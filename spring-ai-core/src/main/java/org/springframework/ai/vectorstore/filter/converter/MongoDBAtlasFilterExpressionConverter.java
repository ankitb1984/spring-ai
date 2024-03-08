package org.springframework.ai.vectorstore.filter.converter;

import org.springframework.ai.vectorstore.filter.Filter;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.AND;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.OR;

/**
 * Converts {@link Filter.Expression} into MongDB Atlas metadata filter expression format.
 * (https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#std-label-vectorSearch-agg-pipeline-filter)
 *
 * @author Chris Smith
 */
public class MongoDBAtlasFilterExpressionConverter extends AbstractFilterExpressionConverter {

	@Override
	protected void doExpression(Filter.Expression expression, StringBuilder context) {
		// Handling AND/OR
		if (AND.equals(expression.type()) || OR.equals(expression.type())) {
			doCompoundExpressionType(expression, context);
		}
		else {
			doSingleExpressionType(expression, context);
		}
	}

	private void doCompoundExpressionType(Filter.Expression expression, StringBuilder context) {
		context.append("{");
		context.append(getOperationSymbol(expression));
		context.append(":[");
		this.convertOperand(expression.left(), context);
		context.append(",");
		this.convertOperand(expression.right(), context);
		context.append("]}");
	}

	private void doSingleExpressionType(Filter.Expression expression, StringBuilder context) {
		context.append("{");
		this.convertOperand(expression.left(), context);
		context.append(":{");
		context.append(getOperationSymbol(expression));
		context.append(":");
		this.convertOperand(expression.right(), context);
		context.append("}}");
	}

	private String getOperationSymbol(Filter.Expression exp) {
		switch (exp.type()) {
			case AND:
				return "$and";
			case OR:
				return "$or";
			case EQ:
				return "$eq";
			case NE:
				return "$ne";
			case LT:
				return "$lt";
			case LTE:
				return "$lte";
			case GT:
				return "$gt";
			case GTE:
				return "$gte";
			case IN:
				return "$in";
			case NIN:
				return "$nin";
			default:
				throw new RuntimeException("Not supported expression type:" + exp.type());
		}
	}

	@Override
	protected void doKey(Filter.Key filterKey, StringBuilder context) {
		var identifier = (hasOuterQuotes(filterKey.key())) ? removeOuterQuotes(filterKey.key()) : filterKey.key();
		context.append("\"metadata." + identifier + "\"");
	}

}
