import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export const EvaluationEmptyState = ({
  title,
  description,
}: {
  title: string;
  description: string;
}) => {
  return (
    <Card className="border-dashed">
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent className="text-muted-foreground text-sm">
        {description}
      </CardContent>
    </Card>
  );
};
