import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

export const EvaluationApiNotice = ({
  title,
  description,
}: {
  title: string;
  description: string;
}) => {
  return (
    <Alert>
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{description}</AlertDescription>
    </Alert>
  );
};
