import { GetEmbeddings} from '@/services/embedding';
import { TypesEmbeddings } from '@/types';
import { useState } from 'react';

export default () => {
  const [embeddings, setEmbeddings] = useState<TypesEmbeddings[]>([]);

  const getEmbeddings = async () => {
    const { data } = await GetEmbeddings();
    setEmbeddings(data);
  };

  return {
    embeddings: embeddings,
    getEmbeddings: getEmbeddings,
  };
}