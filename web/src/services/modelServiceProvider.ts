import { TypesModelServiceProviders } from '@/types';
import { request } from '@umijs/max';


export const GetSupportedModelServiceProviders = (): Promise<{ code: string; data: TypesModelServiceProviders[] }> => {
  return request(`/api/v1/supported_model_service_providers`, {
    method: 'GET',
  });
};

export const GetModelServiceProviders = (): Promise<{ code: string; data: TypesModelServiceProviders[] }> => {
  return request(`/api/v1/model_service_providers`, {
    method: 'GET',
  });
};

export const UpdateModelServiceProvider = (
  provider: string,
  data: TypesModelServiceProviders,
): Promise<{ code: string; data: TypesModelServiceProviders; message?: string }> => {
  return request(`/api/v1/model_service_providers/${provider}`, {
    method: 'PUT',
    data,
  });
};

export const DeleteModelServiceProvider = (
  provider: string,
): Promise<{ code: string; message?: string }> => {
  return request(`/api/v1/model_service_providers/${provider}`, {
    method: 'DELETE',
  });
};