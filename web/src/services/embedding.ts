import { TypesEmbeddings } from '@/types';
import { request } from '@umijs/max';

export const GetEmbeddings = (): Promise<{ code: string; data: TypesEmbeddings[] }> => {
  return request(`/api/v1/available_embeddings`, {
    method: 'GET',
  });
};