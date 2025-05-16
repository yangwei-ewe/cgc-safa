# dataset_partitioner/ds_distributor.dockerfile
FROM node:22-alpine

WORKDIR /app
COPY src/dataset_distributor.js src/
COPY package*.json ./
RUN npm install

EXPOSE 3000

CMD ["node", "src/dataset_distributor.js"]
